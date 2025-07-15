# Voicebot

To run the app, use the following command:

```bash
./start.sh
```

for dev mode:

```bash
./start.sh --dev
```

it will run on
```
http://127.0.0.1:8000/
```

you can view the api with swagger ui on 
```
http://127.0.0.1:8000/docs
```

to setup the environment on Linux run:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

to setup the environment on Windows run:

```cmd 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To use the api points with the locks in front of them, you must get authenticated first.

To get authenticated use the login endpoint to get an access token. Copy the token and scroll to the top of the swagger ui page. Click on the button labled `Authorize` and past the token in the Value field. Now after closing it, you should be able to use all api endpoints.