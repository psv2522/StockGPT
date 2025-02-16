import os
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool
def get_stock_info(symbol: str, key: str):
    """Retrieves detailed information about a stock given its ticker symbol and a specific key.
    If the user asks for the 'price', use 'currentPrice'.
    If the user asks about financial data (e.g., 'revenue', 'earnings'), try to select the appropriate key.
    If the user's request is ambiguous, ask for clarification.

    Common keys: address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio
    If asked generically for 'stock price', use currentPrice
    Mention prices in USD unless otherwise specified, convert if necessary
    """
    try:
        data = yf.Ticker(symbol)
        stock_info = data.info
        value = stock_info[key]
        return str(value)
    except KeyError:
        return f"Invalid key '{key}' for stock '{symbol}'.  Could not retrieve information."
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def get_historical_price(symbol: str, start_date: str = None, end_date: str = None):
    """Fetches historical stock prices.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT).
        start_date: Start date (YYYY-MM-DD).  Defaults to 6 months ago.
        end_date: End date (YYYY-MM-DD). Defaults to today.
    """
    try:
        # Default dates
        if end_date is None:
            end_date = date.today()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        if start_date is None:
            start_date = end_date - timedelta(days=180)  # 6 months ago
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

        if start_date >= end_date:
            return "Start date must be before end date."

        data = yf.Ticker(symbol)
        hist = data.history(start=start_date, end=end_date)
        if hist.empty:
            return f"No historical data found for {symbol} between {start_date} and {end_date}."

        hist = hist.reset_index()
        hist[symbol] = hist['Close']
        return hist[['Date', symbol]]

    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."
    except Exception as e:
        return f"An error occurred: {e}"


def plot_price_over_time(historical_price_dfs):
    """Plots historical stock prices from DataFrames."""
    if not historical_price_dfs:
        return "No data to plot."

    full_df = pd.DataFrame(columns=['Date'])
    for df in historical_price_dfs:
        full_df = full_df.merge(df, on='Date', how='outer')

    fig = go.Figure()
    for column in full_df.columns[1:]:
        fig.add_trace(go.Scatter(x=full_df['Date'], y=full_df[column], mode='lines+markers', name=column))

    fig.update_layout(
        title='Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]),
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.2f',
        xaxis=dict(tickangle=-45, nticks=20, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend_title_text='Stock Symbol',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(bgcolor='white', bordercolor='black')
    )
    filename = "_".join(full_df.columns.tolist()[1:]) + ".png"
    fig.write_image(filename)
    fig.show()
    return f"Plot saved as {filename}"


def call_functions(llm_with_tools, user_prompt):
    """Calls functions based on LLM output, handles multiple tool calls, and plots results."""
    system_prompt = (
        "You are a helpful finance assistant that analyzes stocks and stock prices. "
        "Today is {today}.  "
        "Be concise.  If the user is asking for a comparison, use the get_historical_price tool for each stock. "
        "If no specific dates are mentioned for historical data, default to the last 6 months. "
        "Always use YYYY-MM-DD format for dates."
    ).format(today=date.today())

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    historical_price_dfs = []
    responses = []

    while True:  # Loop for iterative tool calls
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not hasattr(ai_msg, 'tool_calls') or not ai_msg.tool_calls:
            # No more tool calls, break the loop
            if isinstance(ai_msg, AIMessage) and ai_msg.content:
                responses.append(ai_msg.content) # Capture final response
            break

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]  
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]      

            if tool_name == "get_stock_info":
                tool_func = get_stock_info
            elif tool_name == "get_historical_price":
                tool_func = get_historical_price
            else:
                messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id))
                continue

            tool_output = tool_func.invoke(tool_args)
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

            if tool_name == "get_historical_price" and isinstance(tool_output, pd.DataFrame):
                historical_price_dfs.append(tool_output)


    if historical_price_dfs:
      plot_response = plot_price_over_time(historical_price_dfs)
      responses.append(plot_response)

    return "\n".join(responses)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.1,
)

tools = [get_stock_info, get_historical_price]
llm_with_tools = llm.bind_tools(tools)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break
    response = call_functions(llm_with_tools, user_input)
    print("Stock Assistant:", response)