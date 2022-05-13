# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import math
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
from technical.util import resample_to_interval, resampled_merge

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from freqtrade.strategy.hyper import RealParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta

class ScalpingStochEMA(IStrategy):
    # freqtrade hyperopt --strategy StochRSIInActionZone --hyperopt-loss SharpeHyperOptLossDaily --spaces buy stoploss roi  --timerange=20210101-20220501 -e 1000
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    can_short: bool = True


    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
      "0": 0.172,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.333
    use_custom_stoploss = False

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured


    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Number of candles used for calculations in lowest price of period
    # min_price_period: int = 32

    # max loss able for calculation position size
    # max_loss_per_trade = 10 # USD

    # self.buy_long_period.value = 15

    use_exit_signal = True

    risk_of_ruin = 0.03
    # buy_leverage = IntParameter(1, 125, default=3, space='buy', optimize=True)
    # buy_risk_reward_ratio = RealParameter(1, 10, default=2, space='buy', optimize=True)
    # buy_long_period = IntParameter(1, 20, default=4, space='buy', optimize=False)
    # buy_stop_period = IntParameter(3, 40, default=20, space='buy', optimize=False)
    buy_stoch_cross_long = IntParameter(20, 80, default=50, space='buy', optimize=True)
    buy_stoch_cross_short = IntParameter(20, 80, default=50, space='buy', optimize=True)

    buy_ema_slow = IntParameter(15, 200, default=50, space='buy', optimize=True)
    buy_ema_fast = IntParameter(3, 50, default=20, space='buy', optimize=True)
    buy_ema_multiple = IntParameter(2, 20, default=20, space='buy', optimize=True)

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            # 'fastMA': {
            #     'color': 'red',
            #     'fill_to': 'slowMA',
            #     'fill_color': 'rgba(232, 232, 232,0.2)'
            # }, 
            # 'slowMA': {
            #     'color': 'blue',
            # },
            # 'resample_{}_fastMA'.format(4 * buy_long_period.value): {
            #     'color': '#ffccd5',
            # }, 
            # 'resample_{}_slowMA'.format(4 * buy_long_period.value): {
            #     'color': '#89c2d9',
            # },
            # 'lowest': {
            #     'color': '#fff3b0',
            # },
            'stochRSI_d': {
                'color': '#ff6d00',
            },
            'stochRSI_k': {
                'color': '#2962ff',
            }
        },
    }
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # MIN - Lowest value over a specified period
        # for val in self.buy_stop_period.range:
        #     dataframe[f'lowest_{val}'] = ta.MIN(dataframe, timeperiod=val)
        # lowest = ta.MIN(dataframe, timeperiod=self.buy_stop_period.value)
        # dataframe['lowest'] = lowest

        rsi = ta.RSI(dataframe, timeperiod=14)
        #StochRSI
        fastk, fastd = ta.STOCH(rsi, rsi, rsi)
        dataframe['stochRSI_k'] = fastk
        dataframe['stochRSI_d'] = fastd

        frames = [dataframe]
        # for val in self.buy_stop_period.range:
        #     frames.append(DataFrame({
        #         f'lowest_{val}': ta.MIN(dataframe, timeperiod=val)
        #     }))
        for val in self.buy_ema_slow.range:
            frames.append(DataFrame({
                f'ema_slow_{val}': ta.EMA(dataframe, timeperiod=self.buy_ema_multiple.value * val)
            }))
        for val in self.buy_ema_fast.range:
            frames.append(DataFrame({
                f'ema_fast_{val}': ta.EMA(dataframe, timeperiod=self.buy_ema_multiple.value * val)
            }))

        dataframe = pd.concat(frames, axis=1)

        return dataframe

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()

    #     stoploss_price = last_candle[f'lowest_{self.buy_stop_period.value}']

    #     # set stoploss when is new order
    #     if current_profit == 0 and current_time - timedelta(minutes=1) < trade.open_date_utc:
    #     # Convert absolute price to percentage relative to current_rate
    #         return (stoploss_price / current_rate) - 1

    #     return 1 # return a value bigger than the initial stoploss to keep using the initial stoploss

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()

        # stop_price = last_candle['lowest']
        # volume_for_buy = self.max_loss_per_trade / (current_rate - stop_price)
        # use_money = volume_for_buy * current_rate

        # total_balance = self.wallets.get_total('USDT')
        total_balance = 0

        if self.config['stake_amount'] == 'unlimited':
            # Use entire available wallet during favorable conditions when in compounding mode.
            total_balance = max_stake
        else:
            # Compound profits during favorable conditions instead of using a static stake.
            total_balance = self.wallets.get_total_stake_amount()

        max_loss_time = 100 / (self.risk_of_ruin * 100)

        loss = total_balance / max_loss_time

        return loss

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        total_balance = self.wallets.get_total_stake_amount()
        expected_tp = 0;
        if  list(self.minimal_roi.values())[-2] == None :
            expected_tp = list(self.minimal_roi.values())[-1] * 100
        else :
            expected_tp = list(self.minimal_roi.values())[-2] * 100
        expected_sl = self.stoploss * 100;

        

        rr = abs(expected_tp / expected_sl)
        max_loss_time = 100 / (self.risk_of_ruin * 100)

        loss = total_balance / max_loss_time
        profit = loss * rr
        position = loss

        tp_set = profit * 100 / position

        return math.floor(tp_set / expected_tp)

    # def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
    #                 current_profit: float, **kwargs):

    #     rr = abs(current_profit / self.stoploss )
    #     if rr >= self.buy_risk_reward_ratio.value:
    #         return 'tp_rr_above'
        

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # print(dataframe['stochRSI_k'])
        dataframe.loc[
            (
                (dataframe['close'] > dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (dataframe[f'ema_fast_{self.buy_ema_fast.value}'] > dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (qtpylib.crossed_above(dataframe['stochRSI_k'], dataframe['stochRSI_d'])) &  
                (dataframe['stochRSI_k'] < self.buy_stoch_cross_long.value ) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['close'] < dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (dataframe[f'ema_fast_{self.buy_ema_fast.value}'] < dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (qtpylib.crossed_below(dataframe['stochRSI_d'], dataframe['stochRSI_k'])) &  
                (dataframe['stochRSI_k'] > self.buy_stoch_cross_short.value ) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (dataframe[f'ema_fast_{self.buy_ema_fast.value}'] < dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                # (dataframe['stochRSI_k'] > 50 ) &  
                # (qtpylib.crossed_above(dataframe['stochRSI_d'], dataframe['stochRSI_k'])) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['close'] > dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                (dataframe[f'ema_fast_{self.buy_ema_fast.value}'] > dataframe[f'ema_slow_{self.buy_ema_slow.value}']) &
                # (dataframe['stochRSI_k'] > 50 ) &  
                # (qtpylib.crossed_above(dataframe['stochRSI_d'], dataframe['stochRSI_k'])) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        return dataframe


