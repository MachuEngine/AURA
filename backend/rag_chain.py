"""RAG chain: vehicle manual retrieval + LLM synthesis."""
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

VEHICLE_MANUAL = """
[Section 1: Driver Fatigue Management]
The vehicle is equipped with a Driver Monitoring System (DMS). When drowsiness is detected
(yawning, eye closure, head drooping), the system should:
1. Increase cabin air circulation to 'fresh air' mode (external intake).
2. Lower cabin temperature by 2-3°C.
3. Activate seat ventilation at level 2.
4. Play a gentle audio alert at 30% volume.
5. If driving > 4 hours, recommend a rest stop within 2km.

[Section 2: Infant Presence Protocol]
When an infant is detected in the vehicle:
1. Maintain cabin temperature at 22-24°C.
2. Set HVAC fan speed to minimum (level 1).
3. Disable aggressive audio alerts; use gentle chime instead.
4. Enable rear-seat climate zone independently.
5. Reduce maximum volume of media to 40%.

[Section 3: Health Event Response]
On detection of coughing or sneezing by occupants:
1. Switch HVAC to recirculation mode immediately.
2. Activate air purifier at maximum speed.
3. Increase air purifier timer to 15 minutes.
4. Log event with timestamp for health tracking.

[Section 4: Background Noise Calibration]
During elevated ambient background noise:
1. Increase voice recognition sensitivity by 20%.
2. Enable active noise cancellation mode.
3. Reduce notification volume compensation.

[Section 5: HVAC Modes]
HVAC modes available:
- 'internal': recirculation (blocks outside air)
- 'external': fresh air intake from outside
- 'auto': automatic switching based on air quality sensor
Fan speeds: 1 (min), 2, 3, 4 (max)
Temperature range: 16°C - 30°C
"""

CLASS_LABELS = {
    "background_noise": 0,
    "coughing": 1,
    "yawning": 2,
    "infant_crying": 3,
}

LABEL_TO_NAME = {v: k for k, v in CLASS_LABELS.items()}


class MockLLM:
    """Fallback LLM that returns structured mock responses without API calls."""

    def invoke(self, prompt: str) -> str:
        # Parse audio event from prompt
        event = "unknown"
        for label in CLASS_LABELS:
            if label in prompt.lower():
                event = label
                break

        responses = {
            "yawning": (
                "ADVICE: Driver fatigue detected. Switching HVAC to external fresh air mode, "
                "reducing temperature by 2°C, and activating seat ventilation.\n"
                "COMMAND: {\"hvac_mode\": \"external\", \"temperature_delta\": -2, "
                "\"seat_ventilation\": 2, \"alert\": \"gentle_chime\"}"
            ),
            "coughing": (
                "ADVICE: Coughing detected. Activating air purifier and switching to recirculation mode "
                "to prevent external pollutants from entering cabin.\n"
                "COMMAND: {\"hvac_mode\": \"internal\", \"air_purifier\": \"max\", "
                "\"purifier_timer_min\": 15}"
            ),
            "infant_crying": (
                "ADVICE: Infant distress detected. Optimizing cabin environment for infant comfort. "
                "Reducing fan speed and stabilizing temperature.\n"
                "COMMAND: {\"temperature_target\": 23, \"fan_speed\": 1, "
                "\"max_volume_pct\": 40, \"rear_climate\": true}"
            ),
            "background_noise": (
                "ADVICE: Elevated background noise detected. Increasing voice recognition sensitivity "
                "and enabling noise cancellation.\n"
                "COMMAND: {\"voice_sensitivity_boost_pct\": 20, \"anc_mode\": true}"
            ),
            "unknown": (
                "ADVICE: Sound event recorded. Monitoring situation for further action.\n"
                "COMMAND: {\"status\": \"monitoring\"}"
            ),
        }
        return responses.get(event, responses["unknown"])


def build_rag_chain():
    """Build RAG chain with FAISS vectorstore and LLM (with mock fallback)."""
    # Build vectorstore
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([VEHICLE_MANUAL])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Try real LLM, fall back to mock
    llm = _load_llm()

    return retriever, llm


def _load_llm():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("No LLM API key found — using MockLLM fallback")
        return MockLLM()

    try:
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    except Exception as e:
        log.warning(f"Failed to init real LLM ({e}) — using MockLLM fallback")
        return MockLLM()


def run_rag(retriever, llm, audio_event: str, confidence: float,
            driving_hours: float, hvac_status: str) -> dict:
    query = (
        f"Sound event: {audio_event} (confidence: {confidence:.2f}). "
        f"Driver has been driving for {driving_hours:.1f} hours. "
        f"Current HVAC mode: {hvac_status}. "
        f"What immediate vehicle control actions and personalized advice are recommended?"
    )

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = (
        f"You are an intelligent in-vehicle AI assistant.\n\n"
        f"Relevant vehicle manual sections:\n{context}\n\n"
        f"Situation: {query}\n\n"
        f"Provide:\n"
        f"1. ADVICE: Personalized actionable advice for the driver (1-2 sentences)\n"
        f"2. COMMAND: A JSON object with system control parameters\n"
    )

    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    return {
        "query": query,
        "retrieved_context": [d.page_content for d in docs],
        "llm_response": text,
    }
