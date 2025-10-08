# 🧠 Data Science Quiz Bot

An **AI-powered Quiz Application** for testing knowledge in **Statistics, Probability, Machine Learning, and more**.  
Built with **Streamlit (frontend)**, **Flask API (backend)**, and deployed on **AWS ECS Fargate** with a full **CI/CD pipeline (GitHub Actions + Amazon ECR + ECS)**.  

![Quiz Bot Demo](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![AWS ECS](https://img.shields.io/badge/Deployed-AWS%20ECS-orange?logo=amazonaws)
![Docker](https://img.shields.io/badge/Containerized-Docker-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.9-yellow?logo=python)

---

## 📸 Screenshots  

### 🎯 Quiz Interface  
<img src="assets/quiz_ui.png" alt="Quiz UI" width="600"/>

### 📊 Example Question Feedback  
<img src="assets/quiz_feedback.png" alt="Answer Feedback" width="600"/>

---

## 🚀 Features  

✅ Interactive quiz with multiple sections (e.g., Statistics, Probability, Machine Learning).  
✅ AI-powered answer evaluation using **OpenAI GPT models**.  
✅ Real-time feedback (Correct ✅ / Incorrect ❌ / Partial ⚡).  
✅ Frontend in **Streamlit**, backend API with **Flask**.  
✅ Dockerized and deployed using **AWS ECS Fargate**.  
✅ CI/CD pipeline with **GitHub Actions** → automatic build, push, and deploy.  

---

## 🛠️ Tech Stack  

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend API**: Flask (REST API for quiz logic & evaluation)  
- **Containerization**: Docker  
- **Cloud Deployment**: AWS ECS (Fargate) + Amazon ECR  
- **CI/CD**: GitHub Actions  
- **AI/LLM**: OpenAI GPT API  

---

## 📂 Project Structure  
```plaintext
ds-tutor-app/
├── backend/                # Flask API
│   ├── app.py
│   ├── routes/
│   └── ...
├── frontend/               # Streamlit UI
│   └── app.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/      # CI/CD pipelines
│   └── deploy.yml
└── README.md
```
---

## ⚙️ Setup (Local Development)

1️⃣ Clone the repository
   ```bash
   git clone https://github.com/vaibhpande21/ds-tutor-app.git
   cd Data_Science_quiz_bot
   ```
2️⃣ Create a virtual environment
```bash
python -m venv .venv  
source .venv/bin/activate   # Mac/Linux  
.venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set up environment variables
```bash
Create a .env file in the project root:
OPENAI_API_KEY=openai_api_key
```

▶️ Running the App
Start the Streamlit app:
```bash
streamlit run frontend/app.py
```
The app will open in your browser at:
```bash
👉 http://localhost:8501
```

## ▶️ Run the backend (Flask API)
```bash
python backend/app.py
```

🐳 Run with Docker (Locally)
```bash
docker build -t ds_tutor_app .
docker run -p 8501:8501 -p 5001:5001 ds_tutor_app
```

☁️ Deployment on AWS ECS (Fargate)
- ** The project is deployed on AWS ECS Fargate with CI/CD:
- ** GitHub Actions builds & pushes Docker image → Amazon ECR
- ** ECS Task Definition pulls the new image
- ** ECS Service updates the running container with the latest version

🎯 Future Improvements
- ** Add user authentication & scores history
- ** Add more subjects & difficulty levels
- ** Deploy with HTTPS + Load Balancer
- ** Multi-user support (team quizzes, leaderboards)

✨ Author

👨‍💻 Developed by Vaibhav Pandey
📫 Reach me at: 
[LinkedIn](https://www.linkedin.com/in/vaibhav-pandey-re2103/) • [GitHub](https://github.com/vaibhpande21)
 
