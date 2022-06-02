# python3 -m app
# python3 -m streamlit run app/frontend.py --server.port 30002 --server.fileWatcherType none

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.backend:app", host="127.0.0.1", port=8001, reload=True)
