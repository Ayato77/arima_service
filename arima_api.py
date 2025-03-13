from fastapi import FastAPI
import numpy as np
import pmdarima as pm
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list  # histrical data
    steps: int  # prediction steps

@app.post("/predict")
def predict_arima(request: PredictionRequest):
    try:
        forecast = auto_arima(request.data, request.steps)
        return {"forecast": forecast.tolist()}
    except Exception as e:
        return {"error": str(e)}


def auto_arima(ts, steps):
    ts_np = np.array(ts)

    # Auto ARIMA model fitting on training data
    model = pm.auto_arima(
        ts_np,                # training data
        start_p=1,            # initial p (AR term)
        start_q=1,            # initial q (MA term)
        max_p=2,              # maximum p
        max_q=2,              # maximum q
        max_d=1,              # maximum d
        m=1,                  # non-seasonal (m=1 for non-seasonal data)
        d=None,               # automatically select the differencing term
        seasonal=False,       # non-seasonal model
        stepwise=True,        # perform stepwise search for best model
        suppress_warnings=True,
        trace=True            # set to True to see fitting output
    )

    # Print the chosen ARIMA model parameters
    print(f"Selected ARIMA Model: {model.summary()}")

    # Forecast future values
    return model.predict(n_periods=steps)