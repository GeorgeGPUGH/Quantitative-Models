# imports
import yfinance as yf
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import minimize
from scipy.stats import norm
import pandas_market_calendars as mcal

"""
Options Bid/Ask prices only work between 9:30AM ET to 4:00PM ET (NYSE trading hours)
Prices still work at ast 1am GMT, will save dataframe so can use to test in mornings!

Need to consolidate and optimise functions

Edit enterPosition, change to initialInvestment, then use enterPosition to calculate capital expenses for underlying AND options after optimisation

Remove unneeded variables - ensure brevity and flow of code

Add comments as needed

This strategy may be used by a market maker to protect own position and would make profit off the commission of the trades as opposed to pure alpha
"""


def main():
    runtime_start = dt.datetime.now() # Used to track file runtime

    # Display options
    pd.set_option('display.max_columns', None) # Displays all columns
    pd.set_option('display.max_rows', None) # Displays all rows

    # Initialise variables
    stock_ticker = "AAPL"
    interest_rate = 0.055 # Give as decimal, US Treasury interest rate
    total_capital = 100000 # Initial capital in USD
    max_capital_percent = 90 # Percentage of total capital to invest
    no_data_available_log = [] # Log for tracking unavailable data

    # Date range for stock data
    start_date_str = "2023-12-01" # Needs to be in the YYYY-MM-DD format
    end_date_str = dt.date.today().strftime("%Y-%m-%d") # End date is today

    # Dates for expiration range of options
    lower_expiry = "2024-02-05"
    upper_expiry = "2024-10-30"

    # Convert expiration dates to calculate DTE in number of trading days (not calendar days)
    lower_expiry, upper_expiry = tradingDaysConversion(lower_expiry, upper_expiry)

    # Fetch stock data and grab yesterday's closing price
    stock_data = fetchStockData(stock_ticker, start_date_str, end_date_str)
    stock_price = stock_data['Close'].iloc[-1]

    # Upper and lower bound for filtering options around stock price
    upper_bound = stock_price * 1.1
    lower_bound = stock_price * 0.9

    options_data, no_data_available_log = fetchOptionData(stock_ticker, no_data_available_log, upper_bound, lower_bound, upper_expiry, lower_expiry)

    options_data_greeks = getOptionGreeks(options_data, stock_price, interest_rate)
    #print(options_data_greeks)

    sorted_options_data, atm_call_short = sortOptions(options_data_greeks, stock_price)
    #print(atm_call_short)

    num_underlying, capital, initial_delta, initial_gamma, premium_gained = initialPosition(stock_price, total_capital, max_capital_percent, atm_call_short)
    print("Initial Delta: ", initial_delta)
    print("Initial Gamma: ", initial_gamma)

    optimal_multiples, total_delta, total_gamma, cap_invested, num_underlying, capital, cap_options = constructPortfolio(sorted_options_data, initial_delta, num_underlying, initial_gamma, capital, stock_price)
    #print("Optimal multiples for each option:", optimal_multiples)

    print("Amount of shares bought: ",num_underlying," at: $",stock_price,", for a total of: $",cap_invested)
    print("Option shorted with strike: ",atm_call_short['strike'], ", for bid: $", atm_call_short['bid'], "totalling: $", premium_gained)
    print("Options bought for total of: $", cap_options)
    print("Remaining capital: ", capital)

    print("Portfolio Delta:", total_delta)
    print("Portfolio Gamma:", total_gamma)

    # Tracks and prints runtime
    runtime_end = dt.datetime.now()
    print("Runtime: ", runtime_end - runtime_start)

    # Save dataframe
    # sorted_options_data.to_csv('topPercentOptionsData_12_12_23.csv', index=False)


def fetchStockData(stock_ticker, start_date_str, end_date_str):
    """
    Fetches stock data for the specified date range
    """

    stock = yf.Ticker(stock_ticker)
    try:
        stock_data = stock.history(start=start_date_str, end=end_date_str)
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        stock_data = pd.DataFrame()

    # Round to 4 decimal places as that is a great enough resolution for this project
    # Also simplifies the capital calculations later on
    stock_data[['Open', 'High', 'Low', 'Close']] = stock_data[['Open', 'High', 'Low', 'Close']].round(4)

    return stock_data


def fetchOptionData(stock_ticker, no_data_available_log, upper_bound, lower_bound, upper_expiry, lower_expiry):
    """
    Fetches option data for the specified range, and logs any dates for which data is unavailable
    """

    stock = yf.Ticker(stock_ticker)

    # Fetch options data
    options_data = pd.DataFrame()

    try:
        expiries = stock.options
        for exps in expiries:
            
            opt = stock.option_chain(exps)

            if opt.calls.empty and opt.puts.empty:  # Check if there's no data
                no_data_available_log.append(exps)  # Log the date
                continue  # Skip to the next date

            opt = pd.concat([opt.calls, opt.puts])
            opt['expirationDate'] = exps
            options_data = pd.concat([options_data, opt], ignore_index=True)

            # Supposed error in yfinance that gives the wrong expiration dates
            # Code below supposed to fix - see if needed
            options_data['expirationDate'] = pd.to_datetime(options_data['expirationDate']) + dt.timedelta(days = 1)
            # Calculate DTE as a decimal for increase accuracy for Greeks calculations
            options_data['DTE'] = (options_data['expirationDate'] - dt.datetime.today()).dt.days / 365

            # Boolean column if option is a call
            options_data['Call'] = options_data['contractSymbol'].str[4:].apply(
                lambda x: "C" in x
            )

            options_data[['bid', 'ask', 'strike']] = options_data[['bid', 'ask', 'strike']].apply(pd.to_numeric)
            options_data['mark'] = (options_data['bid'] + options_data['ask']) / 2 # Calc mid-point of bid-ask spread

            # Drop unnecessary columnds
            options_data = options_data.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])


            options_data_greeks = options_data[(options_data['strike'] > lower_bound) # Get options data within initial ranges at start of code
                                                 & (options_data['strike'] < upper_bound)
                                                 & (options_data['DTE'] > (lower_expiry / 252))
                                                 & (options_data['DTE'] < (upper_expiry / 252))
                                                 & (options_data['impliedVolatility'] > 0.01)]

            # Remaining: contractSymbol, strike, bid, ask, volume, openInterest, impliedVolatility, expirationDate, DTE, Call, mark

    except Exception as e:
        print(f"Error fetching options data: {e}")

    return options_data_greeks.reset_index(drop=True), no_data_available_log


def tradingDaysConversion(lower_expiry, upper_expiry):
    """
    Converts expiry bounds to trading days
    """
    nyse = mcal.get_calendar('NYSE')

    today_date = dt.datetime.today().date()

    lower_expiry = len(nyse.valid_days(start_date=today_date, end_date=lower_expiry))

    upper_expiry = len(nyse.valid_days(start_date=today_date, end_date=upper_expiry))

    # print(lower_expiry)
    # print(upper_expiry)

    return lower_expiry, upper_expiry


def initialPosition(stock_price, total_capital, max_capital_percent, atm_call_short):

    # Calculate 100 multiple to invest in
    cap_to_invest = (max_capital_percent / 100) * total_capital
    num_underlying = (math.floor(cap_to_invest / stock_price) // 100) * 100 # Find max multiple of 100 underlyng to invest in
    initial_delta = num_underlying

    # Short ATM Call - Arbitrary multiple, will start with 1
    multiple = 1 # How many options to short
    premium_gained = atm_call_short['bid'] * 100 * multiple # Get cost of option
    total_capital += premium_gained # Remove from total_capital
    total_capital = total_capital.round(4) # Round to remove floating point errors
    initial_delta -= atm_call_short['Delta'] * 100 * multiple # Remove option delta from initial delta
    initial_gamma = atm_call_short['Gamma'] * -100 * multiple # Get initial gamma, as option is being sold it is negative
    
    return num_underlying, total_capital, initial_delta, initial_gamma, premium_gained


def sortOptions(options_data_greeks, stock_price):

    # Sort options by liquidity and retrieve top 20% 
    count = int(len(options_data_greeks.sort_values(by='volume', ascending=False)) * 0.2)
    sorted_options_data = (options_data_greeks.sort_values(by='volume', ascending=False)).head(count)

    # Separately, filter out only the rows where 'Call' is True and calculate difference between strike and stock
    call_options = options_data_greeks[options_data_greeks['Call'] == True].copy()
    call_options['strike_diff'] = abs(call_options['strike'] - stock_price)

    # Find the strike price closest to the stock price
    closest_strike = call_options.loc[call_options['strike_diff'].idxmin(), 'strike']

    # Extract all options with the closest strike price
    options_closest_strike = call_options[call_options['strike'] == closest_strike]

    # From these options, find the one with the highest gamma
    atm_call_short = options_closest_strike.sort_values(by='Gamma', ascending=False).iloc[0]
        
    return sorted_options_data, atm_call_short


def getOptionGreeks(options_data_greeks, stock_price, interest_rate):
    """
    Calculates the Greeks for each option in the DataFrame and updates the DataFrame with these values.
    """

    # Iterate over the DataFrame rows
    for index, row in options_data_greeks.iterrows():
        # Extract necessary parameters for each option
        strike_price = row['strike']
        time_to_expiration = row['DTE'] # Leave as a fraction of a year
        impliedVolatility = row['impliedVolatility']
        option_type = 'call' if row['Call'] else 'put'

        # Calculate Greeks
        delta, gamma, vega = calculateGreeks(stock_price, strike_price, interest_rate, time_to_expiration, impliedVolatility, option_type)

        # Store the Greeks in the DataFrame
        options_data_greeks.at[index, 'Delta'] = delta
        options_data_greeks.at[index, 'Gamma'] = gamma
        options_data_greeks.at[index, 'Vega'] = vega

    return options_data_greeks.reset_index(drop=True)


def calculateGreeks(stock_price, strike_price, interest_rate, time_to_expiration, impliedVolatility, option_type='call'):

    # Calculate d1
    d1 = (math.log(stock_price / strike_price) + (interest_rate + 0.5 * impliedVolatility ** 2) * time_to_expiration) / (impliedVolatility * math.sqrt(time_to_expiration))

    if option_type.lower() == 'call':
        # Calculate delta for a call option
        delta = norm.cdf(d1)
    else:
        # Calculate delta for a put option
        delta = -norm.cdf(-d1)

    # Calculate gamma
    gamma = norm.pdf(d1) / (stock_price * impliedVolatility * math.sqrt(time_to_expiration))

    # Calculate vega
    vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiration)

    return delta, gamma, vega


def adjustTotalDelta(total_delta, num_underlying, stock_price, capital):
    # Uses assets to manipulate delta of portfolio

    if total_delta < -0.5 or total_delta > 0.5:
        # Find the closest edge of the range (-0.5 or 0.5)
        closest_edge = -0.5 if total_delta < 0 else 0.5
        # Calculate the difference needed to reach the closest edge
        difference = total_delta - closest_edge
        # The number of integers to add or subtract
        # Use ceil for positive difference and floor for negative difference
        adjustment = -math.ceil(difference) if difference > 0 else -math.floor(difference)

        num_underlying += adjustment # Alter num_underlying
        
        cap_invested, capital = enterPosition(num_underlying, stock_price, capital)

        total_delta += adjustment # Alter total_delta after entering position

        return total_delta, num_underlying, cap_invested, capital
    else:
        # total_delta is within the range, no adjustment needed
        return total_delta


def enterPosition(num_underlying, stock_price, capital):
    cap_invested = (num_underlying * stock_price).round(4)
    capital -= cap_invested # Reduce capital
    capital = capital.round(4) # Round to remove floating point errors
    return cap_invested, capital


def constructPortfolio(sorted_options_data, initial_delta, num_underlying, initial_gamma, capital, stock_price):
    """
    Constructs an initial Delta-Gamma neutral portfolio.
    """

    deltas = sorted_options_data['Delta'].values * 100 # Each option represents 100 shares
    gammas = sorted_options_data['Gamma'].values * 100
    bid_prices = sorted_options_data['bid'].values * 100

    # Store delta and gamma values for plotting
    delta_path = []
    gamma_path = []

    # Objective function: Minimize the absolute combined delta and gamma
    def objective(multiples):
        total_delta = np.sum(deltas * multiples) + initial_delta
        total_gamma = np.sum(gammas * multiples) + initial_gamma
        return abs(total_delta) + abs(total_gamma) * 2000 # Arbitrary coefficient to get better results

    # Initial guess (starting with 0 multiples for each option)
    x0 = np.zeros(len(sorted_options_data))

    # Bounds (assuming you can hold a certain range of multiples for each contract)
    max_multiple = 10  # Define the maximum multiple as per your requirement
    bounds = [(0, max_multiple) for _ in range(len(sorted_options_data))]

    # Perform optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds)

    if result.success:
        # Round the results to get whole number multiples
        optimal_multiples = np.round(result.x)

        # Calculate the total delta and gamma for the portfolio with the optimal multiples
        total_delta = np.sum(deltas * optimal_multiples) + num_underlying
        total_gamma = np.sum(gammas * optimal_multiples) + initial_gamma
        cap_options = np.sum(bid_prices * optimal_multiples)
        capital -= cap_options

        total_delta, num_underlying, cap_invested, capital = adjustTotalDelta(total_delta, num_underlying, stock_price, capital)

        return optimal_multiples, total_delta, total_gamma, cap_invested, num_underlying, capital, cap_options
    else:
        raise ValueError("No solution found")


if __name__ == "__main__":
    main()
