import seaborn as sns
import plotly.express as px

from datathon import Data

data = Data().train.preprocessing()


class Plots:

    def __init__(self, data):
        self.data = data


    # Target distribution
    def target_distribution(self, show=True):
        fig = px.histogram(
            self.data, 
            x="cnt", 
            marginal="box", # or violin, rug
        )
        fig.write_image('plots/target_distribution.png')
        fig.show() if show else ...
        return fig


    # Correlation table
    def correlation_map(self):
        fig = px.imshow(
            self.data.corr().abs(), 
            text_auto=True
        )
        fig.write_image('plots/correlation.png')
        fig.show()
        return fig


Plots(data).correlation_map()
# Temp and atemp 0.99
# month and season 0.865
