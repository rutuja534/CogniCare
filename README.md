# 🧠 CogniCare – AI-Driven CICI Detection & Support Chatbot  

**CogniCare** is an AI-powered healthcare tool designed to help clinicians and patients detect and address **chemotherapy-induced cognitive impairment (CICI)** in cancer survivors.  
It combines a **NeuroImage Prediction Tool** for MRI-based diagnosis with a **Gemini/Groq-powered chatbot** for patient education and support, all accessible via an interactive Gradio interface.  

---

## 🚀 Features  

### 🏥 NeuroImage Prediction Tool  
- CNN model trained on MRI scans to predict **CICI probability**.  
- Preprocessing includes normalization, CLAHE enhancement, denoising, and entropy calculation.  
- Achieves **~83.6% accuracy** on evaluation data.  

### 💬 Cognitive Assessment Chatbot  
- Groq-accelerated inference with LLaMA 2 model.  
- Provides **compassionate, evidence-based guidance** on CICI symptoms, risk factors, and coping strategies.  
- Maintains a **professional yet empathetic tone** for patient comfort.  

### 🌐 Dual Interface  
- **Clinicians**: Real-time neuroimage analysis & prediction.  
- **Patients**: Interactive Q&A and self-assessment support.  

---

## 🛠️ Tech Stack  

| Component                | Technology |
|--------------------------|------------|
| Model Development        | PyTorch, OpenCV, NumPy |
| Chatbot Backend          | Groq API, LLaMA 2, Google Gemini API |
| Frontend / UI            | Gradio, Hugging Face Spaces |
| Data Analysis            | Scikit-learn |
| Deployment               | Hugging Face Spaces / Local |

## 🔒 Security Notes  
- **Never commit API keys** — store them in `.env` files.  
- Ensure `.env` is listed in `.gitignore` to avoid accidental commits.  
- Regenerate your API keys immediately if they have ever been exposed publicly.  
- Groq API and Gemini API keys are **required** for chatbot functionality but must be kept **private**.  

---

## 👨‍⚕️ Disclaimer  
This tool is intended for **educational and research purposes only**.  
It is **not** a substitute for professional medical diagnosis or treatment.  
Always consult a **qualified healthcare provider** for any medical concerns.  


