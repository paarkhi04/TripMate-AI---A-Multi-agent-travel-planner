# ✈️ TripMate AI — Multi-Agent Travel Planner

A production-style **multi-agent travel planning system** built with **LangGraph, LangChain, MCP, Groq, FastAPI, and PostgreSQL**.

TripMate AI uses a Supervisor-driven architecture to dynamically route travel requests across specialized agents for **flights, hotels, weather, budgeting, and itinerary planning**. The system combines **input guardrails, source-grounded responses, Human-in-the-Loop (HITL) approval, feedback-based revision, and persistent LangGraph checkpoints** to create a safe and reviewable travel-planning workflow.

---

## 🚀 Overview

TripMate AI transforms a natural-language travel request into a structured travel plan through a coordinated multi-agent workflow.

Instead of relying on a single LLM response, the system separates responsibilities across specialized agents and uses a **Supervisor Agent** to determine which agents are required for each request.

The workflow supports:

- Dynamic agent selection
- Travel request validation
- Flight schedule lookup
- Hotel information retrieval
- Weather information
- Budget generation using verified data
- Itinerary generation
- Human approval before finalization
- Feedback-driven itinerary revision
- Persistent conversation state using PostgreSQL
- MCP-based external tool integration
- Guardrails against unrelated or unsafe requests

---

## 🧠 Architecture

```text
                         ┌──────────────────────┐
                         │      User Request    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Input Guardrails   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Supervisor Agent   │
                         │ Dynamic Agent Routing│
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌────────────┐     ┌────────────┐     ┌────────────┐
          │Flight Agent│     │ Hotel Agent│     │Weather Agent│
          └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Budget Agent │
                            └──────┬───────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │  Itinerary Agent   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Human Approval (HITL)   │
                       └────────────┬───────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                       Approve              Revise
                          │                   │
                          │            User Feedback
                          │                   │
                          │                   ▼
                          │          Updated Itinerary
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                           ┌─────────────────┐
                           │   Final Agent   │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Final Response │
                           └─────────────────┘

                  ┌────────────────────────────┐
                  │ PostgreSQL Checkpointing   │
                  │ Persistent Graph State     │
                  └────────────────────────────┘
```

---

## 🤖 Agents

### 1. Supervisor Agent

The Supervisor analyzes the user's travel request and dynamically determines which specialist agents are required.

It extracts constraints such as:

- Destination
- Origin
- Travel date
- Duration
- Budget
- Travel style
- Special preferences

The Supervisor then routes the request through the appropriate workflow.

---

### 2. Guardrail Agent

Validates incoming requests before travel planning begins.

It helps prevent the system from processing:

- Non-travel requests
- Unsafe or illegal requests
- Requests that cannot be meaningfully handled as travel plans

---

### 3. Incomplete Request Agent

Handles travel requests that do not contain enough information to begin planning.

For example, when no destination is provided, the system asks the user to provide one instead of triggering unnecessary specialist agents.

---

### 4. Flight Agent

Retrieves flight schedule information through the **AviationStack MCP integration**.

The agent:

- Extracts origin and destination constraints
- Maps supported city names to IATA airport codes
- Filters flight schedules by route
- Handles overnight arrivals
- Uses returned MCP data as the authoritative flight source
- Avoids inventing airfare when ticket pricing is unavailable

---

### 5. Hotel Agent

Retrieves available hotel-related information through the connected travel data sources.

When live hotel information is unavailable, the agent explicitly reports the limitation instead of generating unsupported hotel details or prices.

---

### 6. Weather Agent

Retrieves weather information using a **custom Weather MCP server** backed by OpenWeather.

The custom MCP server provides a domain-specific adapter between the travel workflow and weather data.

---

### 7. Budget Agent

Creates a travel budget using only verified information returned by the specialist agents.

The budget agent follows strict grounding rules:

- Flight prices are used only when explicitly provided
- Hotel prices are used only when explicitly available
- Activity, food, transportation, and miscellaneous costs are not invented
- Unsupported estimates are not presented as factual prices
- Totals are calculated only when the required pricing information is available

---

### 8. Itinerary Agent

Combines the outputs of the specialist agents into a structured travel itinerary.

The agent is instructed to distinguish between:

- Verified information from connected travel tools
- Suggestions that do not require unsupported factual claims
- Information that is unavailable from connected sources

This helps reduce unsupported or hallucinated travel details.

---

### 9. Human Approval Agent — HITL

TripMate AI includes a **Human-in-the-Loop approval stage** before finalization.

The user can:

**Approve**

```text
Draft Itinerary
      ↓
Human Review
      ↓
Approve
      ↓
Final Agent
      ↓
Final Itinerary
```

or request a revision:

```text
Draft Itinerary
      ↓
Human Review
      ↓
User Feedback
      ↓
Revised Itinerary
      ↓
Final Agent
```

This makes the workflow reviewable rather than allowing the AI to finalize every plan automatically.

---

### 10. Final Agent

Produces the final response after the Human-in-the-Loop stage has been completed.

---

## 🔌 MCP Integrations

TripMate AI uses the **Model Context Protocol (MCP)** to connect the agent workflow with external tools and services.

### Tavily MCP

Used for web/search-based travel information through the Tavily MCP integration.

### AviationStack MCP

Used for flight schedule information such as:

- Departure airport
- Arrival airport
- Flight number
- Scheduled departure
- Scheduled arrival
- Airline

The current integration does **not** provide verified ticket prices, so the application does not fabricate airfare.

### Custom Weather MCP Server

A custom MCP server implemented using FastMCP provides weather functionality backed by OpenWeather.

---

## 🛡️ Grounding & Safety

TripMate AI was designed with source-grounded generation in mind.

The system applies rules such as:

- Do not invent flight prices
- Do not invent hotel prices
- Do not fabricate unavailable travel data
- Use specialist-agent outputs as authoritative sources for their respective domains
- Explicitly communicate when required information is unavailable
- Avoid presenting unsupported assumptions as verified facts

The system also uses input guardrails to reject requests that are unrelated to travel or involve unsafe/illegal activity.

---

## 👤 Human-in-the-Loop Workflow

One of the core features of TripMate AI is its approval workflow.

After generating a draft itinerary, the user receives two options:

### ✅ Approve & Generate Final

The draft is approved and passed to the final stage.

### 🔄 Revise Using Feedback

The user provides feedback such as:

```text
Make the itinerary more relaxed and reduce the number of activities.
```

The workflow resumes using the existing LangGraph state and generates a revised plan.

---

## 💾 Persistent State

TripMate AI uses **PostgreSQL checkpointing** through LangGraph's PostgreSQL checkpointer.

This allows the application to maintain graph state across multi-step workflows, including the Human-in-the-Loop approval and revision process.

The application uses a PostgreSQL connection configured through the `DATABASE_URL` environment variable.

---

## 🌐 FastAPI Application

The project includes a FastAPI web application providing:

- Interactive travel planning UI
- Travel planning API
- Human approval API
- Health endpoint
- Thread-based workflow state

### API Endpoints

#### Create Travel Plan

```http
POST /api/travel
```

Example request:

```json
{
  "message": "Plan a 5 day trip to Dubai from Dhaka",
  "thread_id": "optional-thread-id"
}
```

---

#### Approve or Revise Travel Plan

```http
POST /api/travel/approve
```

Example:

```json
{
  "thread_id": "thread-id",
  "approved": false,
  "feedback": "Make the itinerary more relaxed."
}
```

---

#### Health Check

```http
GET /health
```

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| LangGraph | Multi-agent workflow orchestration |
| LangChain | Agent/LLM framework |
| Groq | LLM inference |
| MCP | External tool integration |
| Tavily | Web/search integration |
| AviationStack | Flight schedule data |
| OpenWeather | Weather data |
| FastAPI | Backend API and web application |
| PostgreSQL | Persistent LangGraph checkpoints |
| Jinja2 | HTML templating |
| JavaScript | Frontend interactions |
| CSS | Frontend styling |
| Uvicorn | ASGI server |

---

## 📁 Project Structure

```text
TripMate-AI---A-Multi-agent-travel-planner/
│
├── app.py
├── backend.py
├── mcp_client.py
├── custom_weather_mcp_server.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── LICENSE
├── README.md
│
├── tools/
│   ├── __init__.py
│   ├── flight_tool.py
│   └── tavily_tool.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── script.js
│   └── style.css
│
└── .vscode/
    └── settings.json
```

---

## ⚙️ Prerequisites

Before running the project locally, install:

- Python 3.11+
- Git
- PostgreSQL-compatible database
- Required API credentials

---

## 🚀 Local Setup

### 1. Clone the repository

```powershell
git clone <repository-url>
cd TripMate-AI---A-Multi-agent-travel-planner
```

### 2. Create a virtual environment

Using Python's built-in virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, use Conda or another environment manager.

---

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create a `.env` file in the project root.

Example structure:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DATABASE_URL=your_postgresql_connection_string
```

**Never commit `.env` or expose API credentials publicly.**

---

### 5. Run the application

```powershell
python app.py
```

Or run through Uvicorn:

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

---

## 🧪 Validation & Testing

The workflow was tested across multiple scenarios, including:

- Multi-agent travel planning
- Flight MCP integration
- Weather MCP integration
- Hotel information and fallback handling
- Budget grounding
- Supervisor routing
- Missing destination handling
- Non-travel requests
- Unsafe/illegal requests
- Minimal valid travel requests
- Budget-constrained travel requests
- Specific airline/flight requests
- Human approval
- Human feedback-based itinerary revision

The application also includes syntax validation for the core Python modules before committing changes.

---

## ⚠️ Current Limitations

TripMate AI intentionally avoids generating unsupported information when connected sources do not provide it.

Current limitations include:

- AviationStack flight schedule data does not provide verified ticket prices through the current integration.
- Some hotel information may be unavailable depending on connected data sources.
- External API capabilities depend on the subscription and access level of each provider.
- Travel information is dependent on the availability and accuracy of external services.
- The application should be treated as an AI planning assistant rather than a booking system.

---

## 🔐 Security

API credentials are loaded through environment variables and `.env`.

The repository's `.gitignore` excludes:

```text
.env
.envrc
.venv
env/
venv/
```

Credentials should never be committed to Git or included directly in source code.

---

## 📌 Future Improvements

Potential future enhancements include:

- Live hotel pricing and availability
- Verified airfare pricing
- Hotel booking integration
- Flight booking integration
- More comprehensive airport/city coverage
- User authentication
- Saved travel plans
- Multi-destination trip support
- Improved observability and structured logging
- Cloud deployment
- Automated CI/CD testing

---

## 📄 License

This project follows the license provided in the `LICENSE` file.

---

## 👩‍💻 Author

**Paarkhi Tyagi**

Built as a hands-on project exploring:

- Multi-agent AI systems
- LangGraph workflows
- Model Context Protocol
- LLM orchestration
- Human-in-the-Loop systems
- AI grounding and guardrails
- Stateful agent architectures