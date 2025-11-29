# ☁️ Deploying AIDA to the Cloud

To earn the **Deployment Bonus Points** (and to share your agent with the world), you can deploy AIDA to **Hugging Face Spaces** for free.

## Option 1: One-Click Deployment (Recommended)

1.  **Create a Hugging Face Account:** [Sign up here](https://huggingface.co/join).
2.  **Create a New Space:**
    *   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    *   **Space Name:** `aida-agent`
    *   **License:** `MIT`
    *   **SDK:** `Gradio`
3.  **Upload Files:**
    *   Upload `main.py` and `requirements.txt` to your Space.
    *   *Note:* You can also connect your GitHub repo directly!
4.  **Done!** Your agent is now live on the web.

## Option 2: Google Cloud Run

1.  **Install Google Cloud SDK.**
2.  **Create a `Dockerfile`:**
    ```dockerfile
    FROM python:3.9
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    CMD ["python", "main.py"]
    ```
3.  **Deploy:**
    ```bash
    gcloud run deploy aida --source .
    ```

---

*Note: Since AIDA requires a user-provided API key, it is safe to host publicly. The key is never stored permanently.*
