import requests, json, time, hmac, hashlib
from matplotlib import pyplot as plt
import multiprocessing
import os
from pandas import read_csv
plt.rcParams['figure.figsize'] = (16,4)
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook  
output_notebook()
output_file("prediction.html", title="prediction")

class Bittrex(object):
    # key    : e9f3444bbaf34432bdfc9d75305aa03e 
    # secret : e3138089503b4a3fb53592de439f5646 
    def __init__(self, auth_info=None):
        self.key    = auth_info and auth_info['key']
        self.secret = auth_info and auth_info['secret']
        self.uri    = 'https://bittrex.com/api/v1.1'

    def get_request(self, url, sign):
        return requests.get(url, headers={'apisign': sign}).json()
    
    def infinity(self): 
        while True: 
            yield
    
    def dispatch(self, output, result_only=True, index=None):
        if index:
            return output.json()['result'][index] if result_only else output.json()
        else:
            return output.json()['result'] if result_only else output.json()
        
    
    def get_balance(self, all=False, result_only=True):
        url    = self.uri + '/account/getbalances?apikey={}&nonce={}'.format(self.key, time.time())
        sign   = hmac.new(self.secret.encode(), url.encode(), hashlib.sha512).hexdigest()
        result = self.get_request(url, sign)
        
        if result_only:
            return result['result'] if all else list(filter(lambda x: x['Balance'] > 0, result['result']))
        else:
            return result if all else {'message': result['message'], 
                                   'success': result['success'], 
                                   'result': list(filter(lambda x: x['Balance'] > 0, result['result']))}
    
    def get_summary(self, c='vrm', base='btc', result_only=True):
        return self.dispatch(
            requests.get(self.uri + '/public/getmarketsummary?market={}-{}'.format(base, c)), 
            result_only=result_only, index=0)
        
    
    def tick(self, c='vrm', base='btc', result_only=True):
        return self.dispatch(
            requests.get(self.uri + '/public/getticker?market={}-{}'.format(base, c)),
            result_only=result_only)

    def get_coins(self, base='BTC'):
        return list(map(lambda x: x['MarketCurrency'], filter(lambda y: 
                           y['BaseCurrency'] == base.upper(), 
                           requests.get(self.uri + '/public/getmarkets/').json()['result'])))
    
    
    def get_market(self, result_only=True):
        return self.dispatch(
            requests.get(self.uri + '/public/getmarkets/'),
            result_only)
    
    # all = True for all coins, all = ['ETH', 'VRM', ..] for specific coins
    # loop = 0 for a infinite loop, loop = 5 for 5 loops
    # sleep = 60 for a 60 second sleep time between two requests
    def write_serial(self, all=True, loop=0, sleep=60, folder=''):
        while True:
            try:
                coinlist = self.get_coins() if all == True else all
                loop = self.infinity() if loop == 0 else range(loop)

                for coin in coinlist:
                    coin     = coin.upper()
                    filepath = '{}{}_data.csv'.format(folder, coin)

                    if not os.path.isfile(filepath):

                        f = open(filepath, 'w')
                        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            "timestamp", "tick", "ask", "bid", "vol", "buys", 
                            "sells", "btc_tick", "btc_buys", "btc_sells", "btc_vol", "usdt"))

                    for i in loop:
                        f = open(filepath, 'a')

                        coin_summary = self.get_summary(coin)[0]
                        timestamp    = coin_summary['TimeStamp']
                        tick         = coin_summary['Last']
                        ask          = coin_summary['Ask']
                        bid          = coin_summary['Bid']
                        vol          = coin_summary['BaseVolume']
                        buys         = coin_summary['OpenBuyOrders']
                        sells        = coin_summary['OpenSellOrders']

                        btc_market   = self.get_summary('btc', base='usdt')[0]
                        btc_tick     = btc_market['Last']
                        btc_buys     = btc_market['OpenBuyOrders']
                        btc_sells    = btc_market['OpenSellOrders']
                        btc_vol      = btc_market['Volume']

                        usdt         = tick * btc_tick

                        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            timestamp, tick, ask, bid, vol, buys, sells, btc_tick, btc_buys, btc_sells, btc_vol, usdt))

                        f.close()
                        time.sleep(sleep)
            except:
                time.sleep(30)
    
    def write(self, all=True, loop=0, sleep=60, folder=''):
        coinlist = self.get_coins() if all == True else all        
        
        jobs = []
        for coin in coinlist:
            thread = multiprocessing.Process(target=self.write_serial, args=([coin], loop, sleep, folder))
            jobs.append(thread)
        
        for j in jobs:
            try:
                j.start()
            except:
                j.start()
            
        for j in jobs:
            j.join()

        
        
    def plot(self, c='vrm', base='btc', loop=0, sleep=10):
        try:
            from IPython.display import clear_output
            plt.rcParams['figure.figsize'] = (16,4)
        except:
            raise("This method only works on IPython notebook!")
            
        coin         = c.upper()
        loop         = self.infinity() if loop == 0 else range(loop)
        
        tick, ask, bid = [], [], []
        for i in loop:
            clear_output(wait=True)
            coin_summary = self.get_summary(coin, base=base)[0]
            
            tick.append(coin_summary['Last'])
            ask.append(coin_summary['Ask'])
            bid.append(coin_summary['Bid'])
            
            plt.plot(ask, color='r')
            plt.plot(bid, color='g')
            plt.plot(tick, color='b')
            plt.show()
            
            time.sleep(sleep)
    def plot_file(self, c='vrm', path='data/'):
        data = read_csv('{}{}_data.csv'.format(path,c.upper()), header=0, index_col=0)
        headers       = [i for i in data.axes[1]]
        data = data.values
        

        x = [i for i in range(data.shape[0])]


        plotit = figure(title='analysis', x_axis_label='time (1min per dot)', y_axis_label='value', 
                        width=950, height=300, sizing_mode='scale_width', 
                        tools = "xwheel_zoom,undo,redo,xpan,save,reset,hover",active_scroll="xwheel_zoom")


        plotit.line(x, data[:,0], legend='tick', line_width=2, line_color='blue')
        plotit.line(x, data[:,1], legend='ask', line_width=1, line_color='red')
        plotit.line(x, data[:,2], legend='bid', line_width=1, line_color='green')

        show(plotit)

crypto = Bittrex()
crypto.write(all=True, loop=0, sleep=60, folder='data/')
