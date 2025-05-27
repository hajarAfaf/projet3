#obtenez votre propre authtoken
!ngrok config add-authtoken 2xMR5o9DrgUIkTgnov0xkWOnXbK_Rag4GGbNUJP93DZ7nrwi

#Lancer Streamlit et exposer avec Ngrok
from pyngrok import ngrok
import time
ngrok.kill()

!streamlit run app.py &>/content/logs.txt &

time.sleep(5)
public_url = ngrok.connect("http://localhost:8501")
print(f"🚀 Ton app est accessible ici : {public_url}")
