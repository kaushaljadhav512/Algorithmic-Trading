import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
class Bot:


    def _init_(self, game, nu):
        self.Game = game
        self.numSecurities = nu     
        self.iter = 0  

    def GetBalance(self):
        return self.Game.GetBalance()
    
    def GetPrice(self, security):
        return self.Game.GetPrice(security)

    def BuySecurity(self,security,quantity,price):
        return self.Game.BuySecurity(security,quantity,price)

    def SellSecurity(self,security,quantity,price):
        return self.Game.SellSecurity(security,quantity,price)
    
    def GetInterest(self):
        return self.Game.GetInterest()
    
    def GetBeta(self):
        return self.Game.GetBeta()
    
    def GetShareBalance(self,security):
        return self.Game.GetShareBalance(security)
    
    def GetFeatures(self, security):
        return self.Game.GetFeatures(security)

    def GetMostCorrelated(self, security):
        return self.Game.GetMostCorrelated(security)
    
    def GetLeastCorrelated(self, security):
        return self.Game.GetLeastCorrelated(security)

    try:
        def Run(self):
            for i in range(self.numSecurities):
                p=self.GetPrice(i)
                if self.iter%2==0:
                    self.BuySecurity(i,100,p+10)
                else:
                    self.SellSecurity(i,100,p-10)
            self.iter+=1

        # Generate example time series data
        np.random.seed(42)
        time_steps = np.arange(0, 100)
        data = np.cumsum(np.random.randn(100))  # Replace with your actual securities data

        # Create a pandas DataFrame
        df = pd.DataFrame({'time': time_steps, 'data': data})

        # Split the data into training and testing sets
        train_size = int(0.8 * len(df))
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # Fit ARIMA model
        order = (2, 1, 2)  # ARIMA(p, d, q) order
        model = ARIMA(train_data['data'], order=order)
        fit_model = model.fit(disp=0)

        # Forecast future values
        forecast_steps = len(test_data)
        forecast, stderr, conf_int = fit_model.forecast(steps=forecast_steps)

        # Visualize the results
        plt.figure(figsize=(10, 6))
        plt.plot(train_data['time'], train_data['data'], label='Training Data')
        plt.plot(test_data['time'], test_data['data'], label='Test Data')
        plt.plot(test_data['time'], forecast, label='Forecast', color='red')
        plt.fill_between(
            test_data['time'],
            forecast - 1.96 * stderr,
            forecast + 1.96 * stderr,
            color='r',
            alpha=0.3,
            label='Confidence Interval',
        )
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('ARIMA Forecast')
        plt.legend()
        plt.show()

    except:
        print("exception")
