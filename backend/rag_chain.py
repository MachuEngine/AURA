"""RAG chain: vehicle manual retrieval + LLM synthesis."""
import logging
import os

log = logging.getLogger(__name__)

VEHICLE_MANUAL = """
[Section 1: Sneezing & Coughing — Droplet Spread Prevention]
When repeated sneezing or coughing is detected inside the vehicle:
1. Immediately switch HVAC to recirculation mode (internal) to prevent drawing external air
   and to contain potential droplets within the cabin's air-purifier loop.
2. Activate the air purifier at maximum speed.
3. Set air purifier auto-shutoff timer to 20 minutes.
4. Increase cabin ventilation fan to level 3 for faster air turnover through the purifier.
5. Log the event timestamp for occupant health tracking.
6. If multiple events detected within 5 minutes, trigger a gentle voice notification:
   "Cabin air purification active. Consider stopping for fresh air."

[Section 2: Infant Presence Protocol]
When an infant is detected in the vehicle:
1. Maintain cabin temperature at 22-24°C.
2. Set HVAC fan speed to minimum (level 1).
3. Disable aggressive audio alerts; use gentle chime instead.
4. Enable rear-seat climate zone independently.
5. Reduce maximum volume of media to 40%.

[Section 3: Health Event Response — General]
On detection of any respiratory sound event (coughing, sneezing):
1. Switch HVAC to recirculation mode immediately.
2. Activate air purifier at maximum speed.
3. Increase air purifier timer to 15-20 minutes.
4. Log event with timestamp for health tracking.
5. If driving_hours > 3, recommend rest stop.

[Section 4: Background Noise Calibration]
During elevated ambient background noise (engine, wind, rain):
1. Increase voice recognition sensitivity by 20%.
2. Enable active noise cancellation mode.
3. Reduce notification volume compensation.

[Section 5: HVAC Modes]
HVAC modes available:
- 'internal': recirculation (blocks outside air, routes cabin air through purifier)
- 'external': fresh air intake from outside
- 'auto': automatic switching based on air quality sensor
Fan speeds: 1 (min), 2, 3, 4 (max)
Temperature range: 16°C - 30°C
Air purifier speeds: off, low, medium, max
"""

CLASS_LABELS = {
    "coughing":      0,
    "sneezing":      1,
    "infant_crying": 2,
}

LABEL_TO_NAME = {v: k for k, v in CLASS_LABELS.items()}


class MockLLM:
    """Fallback LLM returning structured responses without any API call."""

    def invoke(self, prompt: str) -> str:
        event = "unknown"
        for label in CLASS_LABELS:
            if label in prompt.lower():
                event = label
                break

        responses = {
            "coughing": (
                "ADVICE: Coughing detected. Activating air purifier and switching to recirculation "
                "mode to prevent external pollutants from entering the cabin.\n"
                "COMMAND: {\"hvac_mode\": \"internal\", \"air_purifier\": \"max\", "
                "\"purifier_timer_min\": 15}"
            ),
            "sneezing": (
                "ADVICE: Sneezing detected. Switching HVAC to internal recirculation immediately "
                "to contain droplets and activating air purifier at maximum speed.\n"
                "COMMAND: {\"hvac_mode\": \"internal\", \"air_purifier\": \"max\", "
                "\"fan_speed\": 3, \"purifier_timer_min\": 20, "
                "\"voice_notification\": \"Cabin air purification active.\"}"
            ),
            "infant_crying": (
                "ADVICE: Infant distress detected. Optimising cabin environment for infant comfort — "
                "stabilising temperature and minimising fan noise.\n"
                "COMMAND: {\"temperature_target\": 23, \"fan_speed\": 1, "
                "\"max_volume_pct\": 40, \"rear_climate\": true}"
            ),
            "unknown": (
                "ADVICE: Sound event recorded. Monitoring situation for further action.\n"
                "COMMAND: {\"status\": \"monitoring\"}"
            ),
        }
        return responses.get(event, responses["unknown"])


def build_rag_chain():
    """Build RAG chain with FAISS vectorstore and LLM (with mock fallback)."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([VEHICLE_MANUAL])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return retriever, _load_llm()


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
        f"What immediate vehicle control actions and personalised advice are recommended?"
    )

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = (
        f"You are an intelligent in-vehicle AI assistant.\n\n"
        f"Relevant vehicle manual sections:\n{context}\n\n"
        f"Situation: {query}\n\n"
        f"Provide:\n"
        f"1. ADVICE: Personalised actionable advice for the driver (1-2 sentences)\n"
        f"2. COMMAND: A JSON object with system control parameters\n"
    )

    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    return {
        "query": query,
        "retrieved_context": [d.page_content for d in docs],
        "llm_response": text,
    }
