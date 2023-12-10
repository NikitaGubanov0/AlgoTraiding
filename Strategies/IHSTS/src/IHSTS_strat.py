import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn_extensions.extreme_learning_machines import GenELMRegressor
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.linear_model import Ridge
import arch
import yfinance as yf
import scipy.stats as ss
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
font = {'family': 'normal',
        # 'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)


# --- Hyper-parameters--- #
ticker = 'VOD'  # '^GSPC'  'AAPL'  'VOD'  'GM'  'NWS'  'VNO'
start_date, end_date = '2010-01-01', '2020-12-31'  # '2010-01-01', '2020-12-31'   '2018-01-01', '2019-12-31'  '2010-01-01', '2020-12-31'
test_date, final_date = datetime.date(2018, 12, 31), datetime.date(2020, 1, 1)
# test_date, final_date = datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)
cash_initial = 500000
learning_window = 800  # 252
backtest_window = 110
step = 150  # 10
level, alpha, beta = 0.05, 0.01, 0.09
flag_backtest = 'train'  # 'train', 'test', 'final'
covid_flag = False


def get_indicators(prices):
    """
        Get data for model (load and process)
    """
    df = pd.DataFrame(index=prices.index)

    # For article testing
    # prices_1 = prices.loc[datetime.date(2014, 10, 14):datetime.date(2014, 10, 14)]
    # prices_2 = prices.loc[datetime.date(2014, 10, 17):datetime.date(2014, 10, 30)]
    # prices_3 = prices.loc[datetime.date(2014, 11, 11):datetime.date(2014, 11, 17)]
    # prices_4 = prices.loc[datetime.date(2014, 11, 21):datetime.date(2014, 11, 21)]
    # prices_5 = prices.loc[datetime.date(2014, 12, 8):datetime.date(2014, 12, 12)]
    # prices_example = pd.concat([prices_1, prices_2, prices_3, prices_4, prices_5])
    # df['Example'] = 0
    # df['Example'].loc[prices_example.index] = 1

    df['Price'] = prices
    df['Return'] = prices.pct_change()
    y = df['Return'].dropna()
    res = arch.arch_model(y=y, mean='AR', lags=1, p=1, o=1, q=1, rescale=True).fit()  # 'GJR-GARCH', dist='studentst'
    df['Filter'] = res.resid

    # fig = res.plot()
    # fig.show()
    # print(res)

    # --- Get indicators --- #
    prices = df['Price']

    # series = df['Filter']  # 'Price'  # TODO!
    series = df['Price']

    # # --- Simple Moving Average (MA) --- # #
    df['MA'] = series.rolling(15).mean()  # ! 15

    # # --- Moving Average Convergence and Divergence (MACD) --- # #
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # # --- Stochastic (K, D) --- # #
    low = series.rolling(14).min()
    high = series.rolling(14).max()
    df['K'] = (series - low) / (high - low) * 100
    df['D'] = (df['K'].shift(2) + df['K'].shift(1) + df['K']) / 3

    # # --- Relative Strength Index (RSI) --- # #
    delta = series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.rolling(14).mean()
    roll_down = down.abs().rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # # --- Larry William's R% (WR) --- # #
    df['WR'] = (high - series) / (high - low)

    # # --- Trend --- # #
    df['MA_P'] = prices.rolling(15).mean()
    # df['MA_P'] = df['MA']

    df['Trend_MA'] = np.nan

    # df['Trend_MA'][((df['MA_P'] > df['MA_P'].shift(1)) & (df['MA_P'] > df['MA_P'].shift(2)))] = 'up'  # !  (series > df['MA']) &
    # df['Trend_MA'][((df['MA_P'] < df['MA_P'].shift(1)) & (df['MA_P'] < df['MA_P'].shift(2)))] = 'down'  # !  (series < df['MA']) &

    # df['Trend_MA'][(series < df['MA']) & ((df['MA'] > df['MA'].shift(1)) & (df['MA'].shift(1) > df['MA'].shift(2))
    #                                       & (df['MA'].shift(2) > df['MA'].shift(3)) & (df['MA'].shift(3) > df['MA'].shift(4)))] = 'up'  # !
    # df['Trend_MA'][(series > df['MA']) & ((df['MA'] < df['MA'].shift(1)) & (df['MA'].shift(1) < df['MA'].shift(2))
    #                                       & (df['MA'].shift(2) < df['MA'].shift(3)) & (df['MA'].shift(3) < df['MA'].shift(4)))] = 'down'  # !

    df['Trend_MA'][(series < df['MA_P']) & ((df['MA_P'] > df['MA_P'].shift(-1)) & (df['MA_P'] > df['MA_P'].shift(-2)) &
                                            (df['MA_P'] > df['MA_P'].shift(-3)) & (df['MA_P'] > df['MA_P'].shift(-4)))] = 'up'  # !
    df['Trend_MA'][(series > df['MA_P']) & ((df['MA_P'] < df['MA_P'].shift(-1)) & (df['MA_P'] < df['MA_P'].shift(-2)) &
                                            (df['MA_P'] < df['MA_P'].shift(-3)) & (df['MA_P'] < df['MA_P'].shift(-4)))] = 'down'  # !

    df['Trend_MA_extended'] = df['Trend_MA'].ffill()
    # df['Trend_MA_extended'] = df['Trend_MA']
    df['Trend_MA'] = df['Trend_MA'].fillna('no')

    # # --- Trading signal  --- # #

    min_prices = prices.rolling(5).min().shift(-2)  # !  .shift(2)
    max_prices = prices.rolling(5).max().shift(-2)  # !  .shift(2)

    # up_signal = (prices - min_prices) / (max_prices - min_prices) * 0.5 + 0.5
    # down_signal = (prices - min_prices) / (max_prices - min_prices) * 0.5

    up_signal = (max_prices - prices) / (max_prices - min_prices) * 0.5 + 0.5
    down_signal = (min_prices - prices) / (max_prices - min_prices) * 0.5 + 0.5
    up_dates = df[df['Trend_MA_extended'] == 'up'].index
    down_dates = df[df['Trend_MA_extended'] == 'down'].index

    df['Trading_signal'] = np.nan
    df['Trading_signal'].loc[up_dates] = up_signal.loc[up_dates]
    df['Trading_signal'].loc[down_dates] = down_signal.loc[down_dates]

    # df['Trading_signal'] = df['Return'].shift(-1)

    # df['Trading_signal_up'] = (max_series - series) / (max_series - min_series)
    # df['Trading_signal_down'] = (min_series - series) / (max_series - min_series)
    # df['Trading_signal'] = df['Trading_signal_up'] + df['Trading_signal_down']

    # df['Trading_signal'] = (series.rolling(5).mean().shift(-5) - series) / series

    # df_example = df.loc[prices_example.index]
    # df = df.dropna()

    df = df.iloc[17:]
    return df


def evaluate_trend(df, test_len):
    """
        Evaluate continious trending value and trend
    """
    X = df[['MA', 'MACD', 'K', 'D', 'RSI', 'WR']]
    y = df['Trading_signal']
    # X = X.merge(X.shift(1), left_index=True, right_index=True, how='outer').iloc[1:]
    # y = y.iloc[1:]

    # Use 4 forward points for getting trend signal, so should exclude last 4 points from train
    y_train, y_test = y.iloc[:-test_len-4].dropna(), y.iloc[-test_len:]
    X_train, X_test = X.loc[y_train.index], X.loc[y_test.index]
    # X_train, X_test = X.iloc[:-test_len-2], X.iloc[-test_len:]

    # --- Normalize --- #
    # X = (X - X.min()) / (X.max() - X.min())
    X_min, X_max = X_train.min(), X_train.max()
    X_train = (X_train - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)

    # y_min, y_max = abs(y_train).min(), abs(y_train).max()
    # y_train = (y_train - y_min) / (y_max - y_min)
    # y_test = (y_test - y_min) / (y_max - y_min)


    nh = 10
    reg = Ridge(alpha=2.0)
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh', random_state=1)
    model = GenELMRegressor(hidden_layer=srhl_tanh, regressor=reg)  # hidden_layer=srhl_tanh,
    model.fit(X_train, y_train)
    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_test_pred = pd.Series(model.predict(X_test), index=y_test.index)

    df['Trading_signal_model'] = y_train_pred.append(y_test_pred)
    y_model_mean = y_train_pred.mean()
    df['Trading_signal_model_mean'] = y_model_mean
    # df['Trading_signal_model_mean'] = 0
    # y_model_mean = 0.5
    df['Trend_model'] = np.nan
    df['Trend_model'][df['Trading_signal_model'] > y_model_mean] = 'up'
    df['Trend_model'][df['Trading_signal_model'] <= y_model_mean] = 'down'

    trend_signal = df['Trend_model'].loc[y_test.index]
    trading_signal = df['Trading_signal_model'].loc[y_test.index]
    trading_signal_mean = df['Trading_signal_model_mean'].loc[y_test.index]

    # trend_signal = df['Trend_MA'].loc[y_test.index]

    return trend_signal, trading_signal, trading_signal_mean


def get_trend(prices, window=learning_window):
    trend = pd.Series()
    trading_signal = pd.Series()
    trading_signal_mean = pd.Series()
    for i in range(window, len(prices) + step, step):
        i_left = i - window
        i_right = i
        if i >= len(prices) - 1:
            i_right = len(prices) - 1
            i_left = i_right - window

        prices_window = prices.iloc[i_left: i_right]
        df = get_indicators(prices_window)
        trend_cur, trading_signal_cur, trading_signal_mean_cur = evaluate_trend(df, step)
        if i >= len(prices) - 1:
            real_step = step - (i - (len(prices) - 1))
            trend_cur, trading_signal_cur, trading_signal_mean_cur = \
                trend_cur.iloc[-real_step:], trading_signal_cur.iloc[-real_step:], trading_signal_mean_cur.iloc[-real_step:]
        trend = trend.append(trend_cur)
        trading_signal = trading_signal.append(trading_signal_cur)
        trading_signal_mean = trading_signal_mean.append(trading_signal_mean_cur)

    return trend, trading_signal, trading_signal_mean


def get_trading_points(trend, trading_signal, trading_signal_mean, prices):
    decision = pd.Series(0, index=trend.index)
    # position = pd.Series(index=trend.index)
    position = 0
    # j = 0
    for j in range(1, len(trend) - 1):
        # if trend[j] != trend[j - 1]:
        if trend[j] == 'up':
            break

    # --- Common sense new --- #
    # ret = prices.pct_change().loc[trend.index]
    # for i in range(j, len(trend) - 1):
    #     if ((position == 0) and (trading_signal.iloc[i] > trading_signal_mean.iloc[i] * 0 + 0.000)) or\
    #        (position == -1) and (ret.iloc[i] > 0):
    #         decision.iloc[i] = 1  # 'BUY'
    #         position = position + 1
    #     elif ((position == 0) and (trading_signal.iloc[i] <= trading_signal_mean.iloc[i] * 0 - 0.000)) or\
    #          (position == 1) and (ret.iloc[i] < 0):
    #         decision.iloc[i] = -1  # 'SELL'
    #         position = position - 1

    # # --- Common sense upgrade --- #
    # for i in range(j, len(trend) - 1):
    #     if ((position == 0) and (trading_signal.iloc[i] > trading_signal_mean.iloc[i] * 1)) or\
    #        ((position == -1) and (trading_signal.iloc[i] > trading_signal_mean.iloc[i] * 1)):
    #         decision.iloc[i] = 1  # 'BUY'
    #         position = position + 1
    #     elif ((position == 0) and (trading_signal.iloc[i] <= trading_signal_mean.iloc[i] * 1)) or\
    #          ((position == 1) and (trading_signal.iloc[i] <= trading_signal_mean.iloc[i]) * 1):
    #         decision.iloc[i] = -1  # 'SELL'
    #         position = position - 1

    # # --- Common sense --- #
    # for i in range(j, len(trend) - 1):
    #     if (position != 1) and (trend.iloc[i] == 'up'):
    #         decision.iloc[i] = 1  # 'BUY'
    #         position = position + 1
    #     elif (position != -1) and (trend.iloc[i] == 'down'):
    #         decision.iloc[i] = -1  # 'SELL'
    #         position = position - 1

    prices = prices.loc[trend.index]
    # --- Article --- #
    last_decision = 0
    for i in range(j, len(trend) - 1):
        # Covid peak bad dates
        if covid_flag:
            if (prices.index[i] > datetime.date(2020, 2, 20)) and (prices.index[i] < datetime.date(2020, 3, 20)):
                continue
        if (position != 1) and (trend.iloc[i] == 'up') and (last_decision != 1):
            decision.iloc[i] = 1  # 'BUY'
            position = position + 1
            last_decision = 1
            last_decision_idx = i
        elif (position != -1) and (trend.iloc[i] == 'down') and (last_decision != -1):
            decision.iloc[i] = -1  # 'SELL'
            position = position - 1
            last_decision = -1
            last_decision_idx = i
        # elif (position == 1) and ((prices.iloc[i] - prices.iloc[last_decision_idx]) / prices.iloc[last_decision_idx] < -0.02):
        #     decision.iloc[i] = -1  # 'SELL'
        #     position = position - 1
        #     last_decision = -1
        #     last_decision_idx = i
        # elif (position == -1) and ((prices.iloc[i] - prices.iloc[last_decision_idx]) / prices.iloc[last_decision_idx] > 0.02):
        #     decision.iloc[i] = 1  # 'BUY'
        #     position = position + 1
        #     last_decision = 1
        #     last_decision_idx = i

    # --- Correct last position --- #
    if position != 0:
        decision.iloc[last_decision_idx] = 0

    # --- Close position --- #
    # if position == -1:
    #     decision.iloc[i] = 1  # 'BUY'
    #     position = position + 1
    # elif position == 1:
    #     decision.iloc[i] = -1  # 'SELL'
    #     position = position - 1

        # Buy and Hold
    # decision[:] = 0
    # decision[0], decision[-1] = 1, -1

    return decision


def get_equity(decision, prices):
    """
        Evaluate pnl of strategy
    """
    zero_day = pd.Series([cash_initial], index=[decision.index[0] - relativedelta(days=1)])
    equity = zero_day.append(pd.Series(index=decision.index))
    cash = pd.Series(index=decision.index)
    vol = pd.Series(index=decision.index)
    pos = pd.Series(index=decision.index)
    cash_cur = cash_initial
    position = 0
    prices = prices.loc[decision.index]
    for date in equity.index[1:]:
        decision_cur, price_cur = decision.loc[date], prices.loc[date]
        # print(date)
        # if date == datetime.date(2020, 2, 24):
        #     tt = 1
        if position == 0:
            vol_cur = (cash_cur // price_cur) * 0.95
        elif position != 0:
            vol_cur = abs(position)

        position = position + decision_cur * vol_cur
        cash_cur = cash_cur - decision_cur * vol_cur * price_cur - 0.001 * abs(decision_cur * vol_cur * price_cur)
        equity_cur = cash_cur + (position * price_cur)  # - cash_initial
        equity.loc[date] = equity_cur
        cash.loc[date] = cash_cur
        vol.loc[date] = vol_cur
        pos.loc[date] = position

        if (equity_cur <= 0) or (cash_cur <= 0):
            raise Exception('Cash is broken at', date)
    return equity, cash, vol, pos


def get_beta(strategy):
    market = yf.download('^GSPC', start_date, end_date)['Adj Close'].pct_change().dropna()
    market = market.loc[strategy.index]
    # treasury = yf.download('^FVX', '2010-01-01', '2020-12-31')['Adj Close'].pct_change().dropna()
    cov_martix = np.cov(strategy, market)
    beta = cov_martix[0, 1] / cov_martix[1, 1]
    return beta


def get_max_drawdown(equity, interval):
    # max_run = np.maximum.accumulate(equity)
    # drawdowns = (max_run - equity) / max_run
    # max_drawdown = np.max(drawdowns)

    max_run = equity.rolling(interval).max()  # .shift(-interval + 1).ffill()
    drawdown = (max_run - equity) / max_run
    return -drawdown.dropna() * 100


def get_sharpe_ratio(ret):
    return ret.mean() / ret.std()


def get_VaR(ret, alpha=alpha):
    return ret.quantile(alpha)


def get_ES(ret, alpha=alpha):
    VaR = get_VaR(ret, alpha)
    return ret[ret < VaR].mean()


def get_rachev_ratio(ret, alpha=alpha, beta=beta):
    # VaR_left = get_VaR(ret, alpha)
    # ETL = ret[ret < VaR_left].mean()
    # VaR_right = -get_VaR(-ret, beta)
    # ETR = ret[-ret < -VaR_right].mean()

    ETL = get_ES(ret, alpha)
    ETR = -get_ES(-ret, beta)

    return abs(ETR / ETL)


def get_marks(ret, equity, VaR_fun, backtest_window=backtest_window, level=level, alpha=alpha, beta=beta):
    VaR = np.full(ret.size, np.nan)
    ES = np.full(ret.size, np.nan)
    Rachev = np.full(ret.size, np.nan)
    VaR_param = np.full(ret.size, np.nan)
    ES_param = np.full(ret.size, np.nan)
    Rachev_param = np.full(ret.size, np.nan)
    Sharpe = np.full(ret.size, np.nan)
    Beta_market = np.full(ret.size, np.nan)
    market = yf.download('^GSPC', start_date, end_date)['Adj Close'].pct_change().dropna()
    # market.index = market.index.to_period(freq='d')
    # treasury = yf.download('^FVX', '2010-01-01', '2020-12-31')['Adj Close'].pct_change().dropna()
    Max_drawdown = np.full(ret.size, np.nan)
    Cumulative_return = np.full(ret.size, np.nan)
    Cum_ret_to_max_drawdown = np.full(ret.size, np.nan)

    equity = equity.loc[equity.index.intersection(ret.index)]

    for i in range(backtest_window, len(ret)):
        history = ret[i - backtest_window: i]


        VaR[i] = history.quantile(level)
        ES[i] = history[history < VaR[i]].mean()

        VaR_left = history.quantile(alpha)
        ES_left = history[history < VaR_left].mean()
        VaR_right = history.quantile(1 - beta)
        ES_right = history[history > VaR_right].mean()
        Rachev[i] = abs(ES_right / ES_left)

        # For parametric evaluation
        VaR_param[i], ES_param[i], Rachev_param[i] = VaR_fun(history, level)

        Sharpe[i] = history.mean() / history.std()

        market_cur = market.loc[market.index.intersection(history.index)]
        cov_martix = np.cov(history, market_cur)
        Beta_market[i] = cov_martix[0, 1] / cov_martix[1, 1]

        equity_cur = equity[i - backtest_window: i]

        Cumulative_return[i] = equity_cur[-1] / equity_cur[0] - 1

        max_run = np.maximum.accumulate(equity_cur)
        drawdowns = (max_run - equity_cur) / max_run
        Max_drawdown[i] = drawdowns.max()

        Cum_ret_to_max_drawdown[i] = Cumulative_return[i] / Max_drawdown[i]

    VaR = pd.Series(data=VaR, index=ret.index)  #  name=VaR_fun.__name__
    ES = pd.Series(data=ES, index=ret.index)
    Rachev = pd.Series(data=Rachev, index=ret.index)
    VaR_param = pd.Series(data=VaR_param, index=ret.index)  #  name=VaR_fun.__name__
    ES_param = pd.Series(data=ES_param, index=ret.index)
    Rachev_param = pd.Series(data=Rachev_param, index=ret.index)
    Sharpe = pd.Series(data=Sharpe, index=ret.index)
    Beta_market = pd.Series(data=Beta_market, index=ret.index)
    Max_drawdown = pd.Series(data=-Max_drawdown * 100, index=ret.index)
    Cumulative_return = pd.Series(data=Cumulative_return * 100, index=ret.index)
    Cum_ret_to_max_drawdown = pd.Series(data=Cum_ret_to_max_drawdown, index=ret.index)#.dropna()

    return VaR, ES, Rachev, VaR_param, ES_param, Rachev_param, Sharpe, Beta_market, Max_drawdown, Cumulative_return, Cum_ret_to_max_drawdown


def calculate_VaR_ES(ret, level=level, alpha=alpha, beta=beta):

    # --- GEV  --- #
    # ret_cur = -ret.copy()
    # ret_cur.index = pd.to_datetime(ret.index)
    # ret_cur.index = ret_cur.index.to_period(freq='D')
    # maximas = ret_cur.resample('W').max()
    # params_genextreme = ss.genextreme.fit(maximas)
    # c, loc, scale = params_genextreme
    # # xi = -c
    # # VaR = loc + scale / xi * (1 - (-5 * np.log(1 - level)) ** (-xi))
    # VaR = ss.genextreme.ppf(1-level, c, loc, scale)
    # ES = ss.genextreme.expect(args=(c,), loc=loc, scale=scale, lb=VaR, conditional=True)
    # VaR_other = ss.genextreme.ppf(level, c, loc, scale)
    # ES_other = ss.genextreme.expect(args=(c,), loc=loc, scale=scale, ub=VaR_other, conditional=True)
    #
    # VaR_left = ss.genextreme.ppf(1-alpha, c, loc, scale)
    # ES_left = - ss.genextreme.expect(args=(c,), loc=loc, scale=scale, ub=VaR_other, conditional=True)
    # VaR_left = - VaR_left
    # VaR_right = ss.genextreme.ppf(beta, c, loc, scale)
    # ES_right = - ss.genextreme.expect(args=(c,), loc=loc, scale=scale, lb=VaR, conditional=True)
    # VaR_right = - VaR_right
    # Rachev = ES_right / ES_left

    # --- t-Student  --- #
    # params_t = ss.t.fit(ret)
    # t_df, t_mean, t_sigma = params_t
    # VaR = ss.t.ppf(alpha, t_df, t_mean, t_sigma)
    # ES = ss.t.expect(args=(t_df,), loc=t_mean, scale=t_sigma, ub=VaR, conditional=True)
    # VaR_left = ss.t.ppf(alpha, t_df, t_mean, t_sigma)
    # ES_left = ss.t.expect(args=(t_df,), loc=t_mean, scale=t_sigma, ub=VaR_left, conditional=True)
    # VaR_right = ss.t.ppf(1-beta, t_df, t_mean, t_sigma)
    # ES_right = ss.t.expect(args=(t_df,), loc=t_mean, scale=t_sigma, lb=VaR_right, conditional=True)
    # Rachev = ES_right / ES_left

    # --- normal  --- #
    loc, scale = ss.norm.fit(ret)
    VaR = ss.norm.ppf(level, loc, scale)
    ES = ss.norm.expect(loc=loc, scale=scale, ub=VaR, conditional=True)
    VaR_left = ss.norm.ppf(alpha, loc, scale)
    ES_left = ss.norm.expect(loc=loc, scale=scale, ub=VaR_left, conditional=True)
    VaR_right = ss.norm.ppf(1-beta, loc, scale)
    ES_right = ss.norm.expect(loc=loc, scale=scale, lb=VaR_right, conditional=True)
    Rachev = abs(ES_right / ES_left)

    return VaR, ES, Rachev


def plot_returns(level=level, alpha=alpha, beta=beta):
    VaR = get_VaR(ret, level)
    ES = get_ES(ret, level)
    VaR_left = get_VaR(ret, alpha)
    ETL = get_ES(ret, alpha)
    VaR_right = -get_VaR(-ret, beta)
    ETR = -get_ES(-ret, beta)

    plt.figure()

    sns.distplot(ret[(ret >= VaR)], hist=True, kde=False,  # & (ret <= VaR_right)
                 # bins=20,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    sns.distplot(ret[ret < VaR], hist=True, kde=False,
                 # bins=5,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4},
                 color='red')

    # sns.distplot(ret[ret > VaR_right], hist=True, kde=False,
    #              bins=20,
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 4},
    #              color='green')

    plt.axvline(x=VaR, linewidth=4, color='red')
    plt.axvline(ES, linewidth=4, color='red', linestyle='dashed')
    plt.axvline(x=VaR_left, linewidth=4, color='orange')
    plt.axvline(ETL, linewidth=4, color='orange', linestyle='dashed')
    plt.axvline(x=VaR_right, linewidth=4, color='green')
    plt.axvline(ETR, linewidth=4, color='green', linestyle='dashed')

    plt.title("Returns distribution", weight="bold")
    plt.legend(['VaR (alpha=' + '{:.2f}%'.format(100 * level) + '):  {:.2f}%'.format(100 * VaR),
                'ES (alpha=' + '{:.2f}%'.format(100 * level) + '):  {:.2f}%'.format(100 * ES),
                'VaR_left (beta=' + '{:.2f}%'.format(100 * alpha) + '):  {:.2f}%'.format(100 * VaR_left),
                'ETL (beta=' + '{:.2f}%'.format(100 * alpha) + '):  {:.2f}%'.format(100 * ETL),
                'VaR_right (beta=' + '{:.2f}%'.format(100 * beta) + '):  {:.2f}%'.format(100 * VaR_right),
                'ETR (beta=' + '{:.2f}%'.format(100 * beta) +'):  {:.2f}%'.format(100 * ETR),
                'Empirical Distribution',
                'Returns < VaR'], fontsize=12)
    plt.show()


# --- Load data --- #
prices = yf.download(ticker, start_date, end_date)['Adj Close']
# trend = get_trend(prices)
prices.index = pd.to_datetime(prices.index).date
# prices.index = prices.index.to_period(freq='d')

if flag_backtest == 'train':
    backtest_date_start, backtest_date_end = datetime.datetime.strptime(start_date, '%Y-%m-%d').date(), test_date
elif flag_backtest == 'test':
    backtest_date_start, backtest_date_end = test_date, final_date
elif flag_backtest == 'final':
    backtest_date_start, backtest_date_end = final_date, datetime.date(2020, 12, 31)

# --- Run model --- #
trend, trading_signal, trading_signal_mean = get_trend(prices)
trend = trend[(trend.index >= backtest_date_start) & (trend.index <= backtest_date_end)]
trading_signal = trading_signal.loc[trend.index]
trading_signal_mean = trading_signal_mean.loc[trend.index]

decision = get_trading_points(trend, trading_signal, trading_signal_mean, prices)

buy_idx, sell_idx = decision[decision == 1].index, decision[decision == -1].index
prices = prices.loc[decision.index]

# --- Plot prices and decision --- #
# fig, ax = plt.subplots(3, 1, sharex=True)
prices.plot(kind='line', label='Price', title=ticker)
sns.scatterplot(data=prices.loc[buy_idx], marker='^', color='green', label='Buy', s=50)
sns.scatterplot(data=prices.loc[sell_idx], marker='v', color='red', label='Sell', s=50)
# trading_signal.plot(kind='line', label='trading_signal', ax=ax[1])
# trend.replace({'up': 1, 'down': -1}).plot(kind='line', label='trend', ax=ax[2])
plt.show()

# --- Plot equity --- #
equity, cash, vol, pos = get_equity(decision, prices)

# fig, ax = plt.subplots(2, 1, sharex=True)
equity.plot(kind='line', label='Equity', title='Equity')
sns.scatterplot(data=equity.loc[buy_idx], marker='^', color='green', label='Buy', s=50)
sns.scatterplot(data=equity.loc[sell_idx], marker='v', color='red', label='Sell', s=50)

# prices.plot(kind='line', label='Price', title=ticker, ax=ax[1])
# sns.scatterplot(data=prices.loc[buy_idx], marker='^', color='green', label='Buy', s=50, ax=ax[1])
# sns.scatterplot(data=prices.loc[sell_idx], marker='v', color='red', label='Sell', s=50, ax=ax[1])
# cash.plot(kind='line', label='Cash', ax=ax[1])
# sns.barplot(x=pos.index, y=pos.values, label='Position', ax=ax[2])
plt.show()

# --- Get cumulative pnl --- #
cumpnl = equity - cash_initial
cumpnl.plot(kind='line', title='Cumulative P&L')
plt.show()

# --- Get cumulative return --- #
cumret = equity / cash_initial * 100  #  - 1
cumret.plot(kind='line', title='Cumulative return, %')
plt.show()

# --- Get return and strategy marks --- #
ret = equity.pct_change().dropna()

VaR, ES, Rachev, VaR_param, ES_param, Rachev_param,\
Sharpe, Beta_market, Max_drawdown, Cumulative_return, Cum_ret_to_max_drawdown = get_marks(ret, equity, calculate_VaR_ES)

# --- Get marks for whole period --- #
VaR_whole = get_VaR(ret, alpha)
ES_whole = get_ES(ret, alpha)
Sharpe_ratio_whole = get_sharpe_ratio(ret)
Rachev_ratio_whole = get_rachev_ratio(ret)

Beta_market_whole = round(get_beta(ret), 3)

max_run_whole = np.maximum.accumulate(equity)
drawdowns_whole = (max_run_whole - equity) / max_run_whole
Max_drawdown_whole = round(np.min(-drawdowns_whole) * 100, 2)

Cumulative_return_whole = round(equity[-1] / equity[0] * 100, 2)

Cum_ret_to_max_drawn_whole = round(Cumulative_return_whole / Max_drawdown_whole, 2)

# --- Plot Risk --- #
ret.plot(kind='line', label='Returns')
VaR.plot(kind='line', label='VaR, level=' + str(level))
ES.plot(kind='line', style='--', label='ES, level=' + str(level))
VaR_param.plot(kind='line', label='VaR_norm, level=' + str(level))
ES_param.plot(kind='line', style='--', label='ES_norm, level=' + str(level))
plt.legend()
plt.show()

# --- Plot Sharpe, Rachev, Beta --- #
fig, ax = plt.subplots(4, 1, sharex=True)
ret.plot(kind='line', label='Returns', ax=ax[0])
ax[0].legend()
Sharpe.plot(kind='line', label='Sharpe ratio', ax=ax[1], color='magenta')
ax[1].legend()
Rachev.plot(kind='line', label='Rachev ratio, alpha=' + str(alpha) + ', beta=' + str(beta), ax=ax[2], color='green')
Rachev_param.plot(kind='line', label='Rachev ratio (norm), alpha=' + str(alpha) + ', beta=' + str(beta), ax=ax[2], color='orange')
ax[2].legend()
Beta_market.plot(kind='line', label='Beta with market (for whole period: ' + str(Beta_market_whole) + ')', ax=ax[3], color='red')
ax[3].legend()
plt.show()


# --- Plot Cum_ret_to_max_drawdown --- #
fig, ax = plt.subplots(4, 1, sharex=True)
ret.plot(kind='line', label='Returns', ax=ax[0])
ax[0].legend()
Cum_ret_to_max_drawdown.plot(kind='line', label='Cumulative return to Max drawdown', ax=ax[1], color='magenta')  # (for whole period: ' + str(Cum_ret_to_max_drawn_whole) + ')'
ax[1].legend()
Cumulative_return.plot(kind='line', label='Cumulative return, %', ax=ax[2], color='green')  # (for whole period: ' + str(Cumulative_return_whole) + ')'
ax[2].legend()
Max_drawdown.plot(kind='line', label='Max drawdown, %', ax=ax[3], color='red')  # (for whole period: ' + str(Max_drawdown_whole) + ')'
ax[3].legend()
plt.show()

# --- Plot max_drawdown (period) --- #
fig, ax = plt.subplots(2, 1, sharex=True)
ret.plot(kind='line', label='Returns', ax=ax[0])
ax[0].legend()
maxdrawdown_day = get_max_drawdown(equity, 2)
maxdrawdown_day.plot(kind='line', style='--', label='Max drawdown (day), %', title='Drawdowns', ax=ax[1])
max_drawdown_week = get_max_drawdown(equity, 7)
max_drawdown_week.plot(kind='line', style='--', label='Max drawdown (week), %', ax=ax[1])
max_drawdown = max_drawdown_week.rolling(7).min().dropna()
max_drawdown.plot(kind='line', label='Max drawdown (week window), %', ax=ax[1])
ax[1].legend()
plt.show()

whole_data = pd.Series([VaR_whole, ES_whole, Sharpe_ratio_whole, Rachev_ratio_whole, Beta_market_whole,
                        Cum_ret_to_max_drawn_whole, Cumulative_return_whole,
                        min(maxdrawdown_day), min(max_drawdown_week), Max_drawdown_whole],
                       index=['VaR_whole', 'ES_whole', 'Sharpe_ratio_whole', 'Rachev_ratio', 'Beta_market_whole',
                              'Cum_ret_to_max_drawn_whole', 'Cumulative_return_whole',
                              'maxdrawdown_day', 'max_drawdown_week', 'Max_drawdown_whole'])
whole_data.to_excel('pic/whole_data.xlsx')

plot_returns()



