---
description: How to deploy the MentalWell app to Streamlit Cloud
---

# Deploying to Streamlit Cloud

Streamlit Cloud is the easiest way to deploy your Streamlit application for free.

## Prerequisites

1.  **GitHub Account**: You need a GitHub account.
2.  **Project on GitHub**: Your project code must be pushed to a GitHub repository.

## Steps

1.  **Prepare your repository**:
    *   Ensure `requirements.txt` is present (It is).
    *   Ensure `packages.txt` is present (I just created it).
    *   Commit and push all your changes to GitHub.

2.  **Sign up/Login to Streamlit Cloud**:
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Sign in with your GitHub account.

3.  **Deploy the App**:
    *   Click **"New app"**.
    *   Select your GitHub repository (`mentalwell_plus-main` or whatever you named it).
    *   Select the branch (usually `main` or `master`).
    *   **Main file path**: Enter `app.py`.
    *   Click **"Deploy!"**.

## Troubleshooting

*   **"ModuleNotFoundError: No module named 'cv2'"**: This usually means `packages.txt` is missing or `opencv-python-headless` is not in `requirements.txt`. We have added `packages.txt` to fix this.
*   **"Memory Error"**: The emotion detection models (Transformers) can be heavy. If the app crashes, we might need to switch to a smaller model or increase resources (if on a paid plan).

## Note on Camera Access

*   Streamlit Cloud runs on a remote server. The `cv2.VideoCapture(0)` in our code tries to access the *server's* camera, which doesn't exist.
*   **CRITICAL**: For a web-deployed app to access *your* (the user's) camera, we need to use a special component like `streamlit-webrtc`.
*   **Current Status**: The current `app.py` uses `cv2.VideoCapture(0)` which works **locally** but will **NOT** work on Streamlit Cloud.
*   **Action Required**: To make it work online, we would need to refactor the camera logic to use `streamlit-webrtc`.

Would you like me to refactor the code to use `streamlit-webrtc` so it works on the web?
