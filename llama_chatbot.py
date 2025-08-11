from groq import Groq
import os
from utils.errors import ChatbotError
import logging
from dotenv import load_dotenv  # Added for environment variable loading

# Load environment variables from a .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

class CICIChatbot:
    def __init__(self, api_key=None):
        try:
            # Use provided API key or get from environment
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            
            if not self.api_key:
                raise ChatbotError("Groq API key not found. Please set GROQ_API_KEY in environment variables.")
            
            # Initialize Groq client
            self.client = Groq(api_key=self.api_key)
            logger.info("Chatbot initialized successfully with Groq")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise ChatbotError(f"Failed to initialize chatbot: {str(e)}")
    
    def get_response(self, patient_input):
        try:
            if not self.client:
                raise ChatbotError("Chatbot not properly initialized")
            
            if not patient_input or not isinstance(patient_input, str):
                raise ChatbotError("Invalid input message")
            
            if len(patient_input.strip()) == 0:
                raise ChatbotError("Empty message")
            
            # Create chat completion
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a supportive medical assistant specializing in 
                        chemotherapy-induced cognitive impairment (CICI). Your role is to:
                        1. Provide accurate, evidence-based information about CICI
                        2. Explain symptoms, risk factors, and management strategies
                        3. Offer emotional support and practical advice
                        4. Always maintain a compassionate and professional tone
                        5. Remind users to consult healthcare professionals for medical advice"""
                    },
                    {
                        "role": "user",
                        "content": patient_input
                    }
                ],
                model="llama2-70b-4096",
                temperature=0.7,
                max_tokens=1000
            )
            
            response = chat_completion.choices[0].message.content
            logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            logger.error(f"Chatbot error: {str(e)}")
            return (
                "I apologize, but I'm having technical difficulties. "
                "Please try again later or contact support."
            )
