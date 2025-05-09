# The-Price-Is-Right
An Autonomous Agentic AI System for Real-Time Deal Discovery- a modular, multi-agent AI framework that autonomously discovers, evaluates, and pushes online deals to users—powered by LLMs, traditional ML, vector databases, and real-time orchestration. 

Here’s a detailed technical breakdown of each agent in this project, along with the specific tools, models, and technologies used by each. This will serve as a foundation for showing how every component contributes to the autonomous agentic AI system.
________________________________________
🧠 Complete Agent Architecture of "The Price Is Right"

![image](https://github.com/user-attachments/assets/e9663385-d3df-4ef7-8754-6dd2da97a1da)

________________________________________
# 1. 🧭 Planning Agent – The Orchestrator

Role:
Coordinates all other agents. It initiates the system, triggers scanning, processes deals, and decides whether to notify the user.

Workflow Responsibilities:

•	Calls the ScannerAgent to fetch new deals

•	Uses EnsembleAgent to estimate prices

•	Selects the best opportunity (based on discount)

•	If discount > $50, uses MessagingAgent to alert user

Tech Stack:

•	Python OOP for agent design

•	Modular design to enable swappable pricing or messaging agents
________________________________________
# 2. 🔎 Scanner Agent – The Intelligent Scraper

Role:
Finds new deals from RSS feeds and selects the best 5 using LLMs with structured output.

How it works:

•	Scrapes RSS feeds using ScrapedDeal.fetch()

•	Constructs a structured JSON prompt using clear product criteria

•	Sends the prompt to OpenAI's GPT-4o-mini

•	Uses response_format=DealSelection to strictly enforce valid JSON

Tools & Tech Stack:

•	OpenAI API (GPT-4o-mini)

•	Structured Output parsing for JSON validation

•	Natural Language Processing for summarizing and rephrasing product info

•	Logging + filtering logic to avoid duplicate entries
________________________________________
# 3. 📊 Ensemble Agent – The Value Synthesizer

Role:
Combines predictions from three different pricing models and returns a weighted, aggregated price.

Workflow:
•	Calls:

o	SpecialistAgent (LLM)

o	FrontierAgent (LLM + RAG)

o	RandomForestAgent (ML model)

•	Collects prices from all three

•	Creates a feature vector including Min/Max

•	Runs a Linear Regression model (trained separately)

•	Returns weighted predicted price

Tech Stack:

•	Scikit-learn for ensemble regression (ensemble_model.pkl)

•	Pandas for data formatting

•	Joblib for model loading

•	Modular agent integration
________________________________________
# 4. 🔬 Specialist Agent – The Fine-Tuned LLM Model

Role:
Predicts product price using a fine-tuned Meta-LLaMA 3.1-8B model hosted on Modal.

Details:

•	Hosted remotely using Modal's GPU infrastructure

•	Model is quantized and fine-tuned with PEFT for efficiency

•	Uses a custom pricing prompt, e.g., "How much does this cost?"

•	Returns structured price output

Tech Stack:

•	Modal for serverless deployment (modal.App)

•	Transformers + PEFT (Parameter-Efficient Fine-Tuning)

•	Quantization (bitsandbytes) for fast inference

•	Meta-LLaMA 3.1-8B base model, fine-tuned for price estimation

•	Hugging Face Hub for model download & storage
________________________________________
# 5. 🧠 Frontier Agent – The RAG-Powered Estimator

Role:
Estimates price using Retrieval-Augmented Generation with similar products retrieved from a vector database.

How it works:

•	Encodes the product description with Sentence Transformers

•	Queries ChromaDB to retrieve 5 similar products

•	Builds a detailed prompt with price context

•	Sends to GPT-4o-mini or DeepSeek LLM

•	Parses price from LLM response

Tech Stack:

•	ChromaDB for vector storage and retrieval

•	Sentence Transformers (all-MiniLM-L6-v2) for embeddings

•	OpenAI GPT-4o-mini or DeepSeek for LLM predictions

•	Prompt engineering for multi-example context injection

•	Regex parsing to extract numeric price from LLM output
________________________________________
# 6. 🌲 Random Forest Agent – The Classic ML Baseline

Role:
Predicts price using a trained Random Forest model on semantic embeddings of product descriptions.

Workflow:

•	Encodes description using SentenceTransformers

•	Loads pre-trained model (random_forest_model.pkl)

•	Predicts price and returns the result

Tech Stack:

•	Scikit-learn Random Forest Regressor

•	SentenceTransformers (all-MiniLM-L6-v2)

•	Joblib for model serialization
________________________________________
# 7. 📢 Messaging Agent – The Notifier

Role:
Sends push notifications (via Pushover) or SMS alerts (via Twilio – optional).

Workflow:

•	Constructs a concise message with price, estimate, discount, and product URL

•	Sends either:

o	Push Notification using Pushover API

o	SMS using Twilio API (currently disabled by default)

Tech Stack:

•	Pushover API (for real-time notifications)

•	Twilio (optional, SMS-ready)

•	Environment Variables (.env) for API keys and credentials

•	HTTP client (http.client) and URL encoding (urllib)
________________________________________
# 🧩 Bonus: UI Layer (Gradio)

Role:
Offers a user-facing interface for live interaction and deal display.

Features:

•	Live table of deals: Description, Price, Estimate, Discount, URL

•	Periodic refresh with gr.Timer

•	Deal selection triggers alerts

•	Fully local and browser-based launch

Tech Stack:

•	Gradio Blocks

•	Python event-driven handlers

•	Live Refresh with gr.Timer
________________________________________

# 🚀 How to Run the Project

You can run the project using:

python price_is_right_final.py 

or 

python price_is_right.py
________________________________________

Note: Model files (ensemble_model.pkl, random_forest_model.pkl, test.pkl, train.pkl) and vector store (products_vectorstore) are excluded due to size contraints.
