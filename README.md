# Voicebot

To run the app, use the following command:

#### Server IP: `172.17.180.124`

```bash
./start.sh
```

For dev mode:

```bash
./start.sh --dev
```

It will run on:
```bash
http://127.0.0.1:8000/
```

You can view the API with Swagger UI at:
```bash
http://127.0.0.1:8000/docs
```

To set up the environment on Linux, run:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

To set up the environment on Windows, run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Testing the API
To use the API endpoints with the locks in front of them, you must get authenticated first.

To get authenticated, use the login endpoint to get an access token. Copy the token and scroll to the top of the Swagger UI page. Click on the button labeled `Authorize` and paste the token in the Value field. After closing it, you should be able to use all API endpoints.


### Audio to Audio Pipeline
Since the audio streaming is done through WebSockets instead of HTTPS, it cannot be tested through the API.

For this purpose, I have made a `test_client.py` file to mimic the frontend for testing purposes.

To use the client, start the app using the start.sh as stated above and then run the client by running the command:

```python test_client.py```

You will need to replace the `localhost` with `172.17.180.124` when access the test client from an environment different to the server.


The text query response will be printed on the terminal and the audio output will be saved in the folder `backend/rag/temp_audioFiles` with a name like `tts_{timestamp}_{task_id}.wav`.
You can paste the given file name in the fast api tts/download endpoint to access the response remotely.

**Note**: If the program does not stop after pressing Ctrl+C, open a new terminal in the same directory and run the following command:

```./stop.sh```

If it still does not work, run the following command:

```redis-cli -n 0 FLUSHDB && redis-cli -n 1 FLUSHDB```
