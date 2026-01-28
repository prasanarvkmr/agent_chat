# Contact Center AI Agent - Outage Prediction & Operations Intelligence

An AI-powered data analyst agent for contact center operations. Predicts outages, analyzes operational data from multiple systems (Genesys, IVR, Incidents, Infrastructure), and provides actionable insights through natural language queries.

## 🎯 Key Capabilities

- 🔮 **Outage Prediction**: Calculate risk scores using leading indicators
- 📊 **Multi-Source Analysis**: Genesys, Nuance IVR, Watson IVR, Incidents, Solarwinds, Nexthink, ThousandEyes
- 🤖 **Intelligent Data Discovery**: Handles undocumented schemas automatically
- 📈 **Leading Indicators**: Real-time calculation of warning thresholds
- 🔍 **Anomaly Detection**: Identifies unusual patterns before they become outages
- 💡 **Actionable Recommendations**: Suggests interventions based on data analysis
- 🔗 **Azure Databricks**: Direct connection to your data warehouse
- ⚡ **Fast Startup Option**: Disable metadata cache for quicker initialization

## 📖 Documentation

**See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete flowcharts and beginner guide.**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (copy and edit .env)
cp .env.example .env

# 3. Run the application
python app.py

# 4. Open browser: http://localhost:8000
```

## ⚙️ Configuration

### Environment Variables

```env
# LLM Configuration (via Kong Gateway)
KONG_AI_GATEWAY_BASE_URL=https://your-kong-gateway.com
KONG_API_KEY=your-api-key
LLM_MODEL=gpt-4  # Options: gpt-4, claude-sonnet, gemini-pro

# Azure Databricks
DATABRICKS_SERVER_HOSTNAME=your-workspace.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_ACCESS_TOKEN=your-token
DATABRICKS_CATALOG=your_catalog
DATABRICKS_SCHEMA=your_schema

# Performance Options
USE_METADATA_CACHE=true          # Set 'false' for faster startup
METADATA_CACHE_REFRESH_HOURS=24
```

### Disabling Metadata Cache (Faster Startup)

If startup is slow due to metadata loading:

```bash
# Option 1: Environment variable
USE_METADATA_CACHE=false

# Option 2: In code
from agents.data_agent import create_data_agent
agent = create_data_agent(use_metadata_cache=False)
```

## 💬 Example Queries

| Use Case | Example Query |
|----------|---------------|
| Health Check | "Analyze contact center health for the last 4 hours" |
| Outage Risk | "Calculate current outage risk with leading indicators" |
| Call Analysis | "Show call volume trends and abandonment rates" |
| IVR Issues | "Check IVR recognition rates and containment metrics" |
| Correlation | "Correlate recent incidents with operational metrics" |
| Custom | "Find network latency data and detect anomalies" |

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Web UI        │────▶│   FastAPI       │────▶│   AI Agent      │
│   (Business)    │     │   Backend       │     │   (Google ADK)  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌──────────────────────────┴───────┐
                              │                                  │
                              ▼                                  ▼
                    ┌─────────────────┐              ┌─────────────────┐
                    │  Kong AI Gateway│              │ Azure Databricks│
                    │  (GPT-5/Claude/ │              │ SQL Warehouse   │
                    │   Gemini)       │              │                 │
                    └─────────────────┘              └─────────────────┘
```

## How It Works

1. **User asks a question** (e.g., "How is the system health?")
2. **Agent discovers data** - Lists tables and examines their structure
3. **Agent identifies health indicators** - Finds timestamp columns, status fields, metrics
4. **Agent analyzes data** - Queries recent data, detects patterns and anomalies
5. **Agent interprets results** - Uses LLM to understand what the data means
6. **Agent generates report** - Provides health status, KPIs, and recommendations

## Project Structure

```
agent_chat/
├── app.py                 # FastAPI application (main entry point)
├── agent.py               # AI Agent configuration using Google ADK
├── tools.py               # Agent tools for Databricks operations
├── databricks_client.py   # Databricks connection and query handling
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables
├── templates/
│   └── index.html         # Chat UI template
└── README.md              # This file
```

## Prerequisites

1. **Python 3.10+**
2. **Azure Databricks** workspace with SQL Warehouse
3. **OpenAI API key**

## Setup

### 1. Clone and Install Dependencies

```bash
cd agent_chat
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4

# Azure Databricks Configuration
DATABRICKS_SERVER_HOSTNAME=your-workspace.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_ACCESS_TOKEN=dapi-your-token-here

# Default catalog and schema
DATABRICKS_CATALOG=your_catalog
DATABRICKS_SCHEMA=your_schema
```

### 3. Get Databricks Connection Details

1. Go to your Azure Databricks workspace
2. Navigate to **SQL Warehouses**
3. Select your warehouse and click **Connection details**
4. Copy the **Server hostname** and **HTTP path**
5. Generate a **Personal Access Token** from User Settings > Developer > Access tokens

## Running the Application

```bash
python app.py
```

The application will start at `http://localhost:8000`

## Usage

1. Open your browser to `http://localhost:8000`
2. Type a question about your data in the chat input
3. The AI agent will:
   - Understand your question
   - List available tables if needed
   - Examine table schemas
   - Write and execute SQL queries
   - Return results in a friendly format

### Example Questions

Business users can ask natural language questions like:

- "How is the overall system health?"
- "Are there any critical issues right now?"
- "Show me the health status for the last 24 hours"
- "What are the key KPIs I should know about?"
- "Are there any anomalies or unusual patterns?"
- "Give me recommendations to improve system performance"
- "Analyze system health for the past week"
- "What errors are occurring most frequently?"

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/api/chat` | POST | Send message to AI agent |
| `/api/clear` | POST | Clear conversation history |
| `/api/health` | GET | Health check and status |
| `/api/tables` | GET | List available tables |

## Troubleshooting

### Connection Issues

1. **Check Databricks credentials**: Verify your server hostname, HTTP path, and access token
2. **Network access**: Ensure your machine can reach the Databricks workspace
3. **Token expiry**: Personal access tokens expire - generate a new one if needed

### Query Errors

1. **Table not found**: Make sure the catalog and schema are configured correctly
2. **Permission denied**: Verify your Databricks user has access to the tables

### LLM Issues

1. **Rate limits**: If you hit OpenAI rate limits, wait a moment and try again
2. **Invalid API key**: Double-check your OpenAI API key

## Customization

### Adding New Tools

Edit `tools.py` to add new capabilities:

```python
def my_custom_tool(param: str) -> dict:
    """Your custom tool logic."""
    return {"result": "..."}

my_tool = FunctionTool(
    name="my_tool",
    description="Description for the LLM",
    func=my_custom_tool
)

all_tools.append(my_tool)
```

### Modifying the System Prompt

Edit the `SYSTEM_PROMPT` in `agent.py` to change the agent's behavior.

### Changing the UI

Modify `templates/index.html` to customize the chat interface.

## Security Considerations

- Never commit `.env` files with real credentials
- Use Databricks service principals in production
- Implement proper authentication for the web UI in production
- The agent is configured for READ-ONLY operations by default

## License

MIT License
