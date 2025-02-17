# FastAPI Hello World (Production mode, Deploy mode)

## Getting Started

```
float16 example fastapi_helloworld
float16 deploy app.py
```

Function mode : Best for simple, stateless, and short-lived functions.
```
curl -X GET https://api.float16.cloud/task/run/function/<your_function_endpoint> \
-h "Content-Type: application/json" \
-h "Authorization Bearer <endpoint_token>"
```

Server mode : Best for low latency, high throughput and cost-effective.
```
curl -X GET https://api.float16.cloud/task/run/server/<your_function_endpoint> \
-h "Content-Type: application/json" \
-h "Authorization Bearer <endpoint_token>"
```

## Description

This is a simple FastAPI example that returns "Hello World".

## High Level Overview

```
+-----------------+    +-----------------+
|                 |    |                 |
|  FastAPI        |<---|  User           |
|                 |    |                 |
+-----------------+    +-----------------+
         |
         | GET /
         |
         v
+-----------------+
|                 |
|  Return         |
|  "Hello World"  |
|                 |
+-----------------+
```

## Libraries 

- fastapi==0.104.1
- uvicorn==0.23.0

## GPU Configuration

- L4 

## Expected Performance

Function mode performance:
- Return "Hello World" in less than 1 sec

Server mode performance:
- Return "Hello World" in less than 500 ms

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No