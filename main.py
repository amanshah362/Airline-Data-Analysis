import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('airlines_flights_data.csv')


# Understanding the Behaviour of Data
print(df.head())
print("\nData Shape")
print(df.shape)

# Checking the Missing Values
print(df.isnull().sum())

# Cleaning The Data
print("Cleaning Data....")
df.drop_duplicates(inplace=True)

# Detectinng Outiers
columns_to_check = ["price", "duration"]

for col in columns_to_check:
    print(f"\nCompute Outliers of {col}")

    # Convert column to numeric, coerce errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in this column (optional: depending on your use case)
    df_clean = df[df[col].notna()]

    # Compute outliers
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    print("\nNumber of outliers:", outliers.sum())
    
    
# Data Analysis

# Step - 1 Calculate Average Ticket Price by Airlines
print("\nAverage Ticket Price by Airlines")
avg_ticket_price = df.groupby("airline")["price"].mean()
print(avg_ticket_price)

# Step - 2 Calculate Average Flight Duration by Airlines
print("\nAverage Flight Duration by Airlines")
avg_flight_duration = df.groupby("airline")["duration"].mean()
print(avg_flight_duration)

# Step - 3 Calculate Total Flight Count by Airlines
print("\nTotal Flight Count by Airlines")
total_flight_count = df.groupby("airline")["flight"].count()
print(total_flight_count)

# Step - 4 Calculate Ticket Price Variation by Day left Before Departure
print("\nTicket Price Variation by Day left Before Departure")
ticket_price_variation = df.groupby("days_left")["price"].var()
print(ticket_price_variation)

# Step - 5 Compare Price between Economy and Business Class
print("\nCompare Price between Economy and Business Class")
economy_price = df[df["class"] == "Economy"]["price"].mean()
business_price = df[df["class"] == "Business"]["price"].mean()
print(f"Economy Class Price: {economy_price}")
print(f"Business Class Price: {business_price}")

# Step - 6 Identify most popular source-destination city pairs
print("\nIdentify most popular source-destination city pairs")
popular_pairs = df.groupby(["source_city", "destination_city"])["flight"].count().sort_values(ascending=False).head(5)
print(popular_pairs)

# Step - 7 Airline Domain on Secific Routes
print("\nAirline Domain on Secific Routes")
airline_domain = df.groupby(["airline", "source_city", "destination_city"])["flight"].count().sort_values(ascending=False).head(5)
print(airline_domain)

# Step - 8 Count flight frequency per source city
print("\nCount flight frequency per source city")
flight_frequency = df["source_city"].value_counts()
print(flight_frequency)

# Step - 9 Examine Price Changes Based on Departure
print("\nExamine Price Changes Based on Departure")
price_changes = df.groupby("arrival_time")["price"].var()
print(price_changes)

# Step - 10 Calculate Average Duration route
print("\nCalculate Average Duration route")
avg_duration = df.groupby(["source_city", "destination_city"])["duration"].mean()
print(avg_duration)

# Step - 11 How Number of Stops Affects Price
print("\nHow Number of Stops Affect Price")
stops_price = df.groupby("stops")["price"].mean()

print(stops_price)

# Step - 12 How Number of Stops Affects Duration
print("\nHow Number of Stops Affect Duration")
stops_duration = df.groupby("stops")["duration"].mean()
print(stops_duration)

# Step - 13 Identify Airlines Average Fares
print("\nIdentify Airlines Average Fares")
airline_fares = df.groupby("airline")["price"].mean()
print(airline_fares)

# Step - 14 Identify Airlines Average longest Shortest Duration
print("\nIdentify Airlines Average longest Shortest Duration")
airline_duration = df.groupby("airline")["duration"].mean()
print(airline_duration)

# Step - 15 Calculate Airline Market Share Based on Flight Count
print("\nCalculate Airline Market Share Based on Flight Count")
airline_market_share = df.groupby("airline")["flight"].count() / df["flight"].count()
print(airline_market_share)


# Plotting


fig, ax = plt.subplots(2, 4, figsize=(26, 12))
ax = ax.flatten()

# Plot - 1 Bar Chart Average ticket per Airline
avg_ticket = df.groupby("airline")["price"].mean()
ax[0].bar(avg_ticket.index, avg_ticket.values, color='green')
ax[0].set_title("Average Ticket Price by Airlines")
ax[0].set_xlabel("Airline")
ax[0].set_ylabel("Average Ticket Price")
ax[0].tick_params(axis='x', rotation=45)


# Plot - 2 Bar Chart Ticket Price vs days left before departure
days_left = df.groupby("days_left")["price"].mean()
ax[1].barh(days_left.index, days_left.values, color='red')
ax[1].set_title("Ticket Price vs days left before departure")
ax[1].set_xlabel("Average Ticket Price")
ax[1].set_ylabel("Days Left Before Departure")



# Plot - 3 Bar Chart Average Flight Duration by Airlines
avg_duration = df.groupby("airline")["duration"].mean()
ax[2].bar(avg_duration.index, avg_duration.values, color='blue')
ax[2].set_title("Average Flight Duration by Airlines")
ax[2].set_xlabel("Airline")
ax[2].set_ylabel("Average Flight Duration")
ax[2].tick_params(axis='x', rotation=45)

# Plot - 4 pie Chart Market share by flight count
airline_market_share = df.groupby("airline")["flight"].count() / df["flight"].count()
ax[3].pie(airline_market_share, labels=airline_market_share.index, autopct='%1.1f%%', startangle=90)
ax[3].set_title("Market Share by Flight Count")

# Plot - 5 Box Plot Stops vs price
ax[4].boxplot(df["price"], labels=["Price"])
ax[4].set_title("Price vs Stops")
ax[4].set_xlabel("Stops")
ax[4].set_ylabel("Price")

# Plot - 6 Stacked Bar Chart Airline dominance on routes
airline_dominance = df.groupby(["airline", "source_city", "destination_city"])["flight"].count().unstack(level=0).head(10)
airline_dominance.plot(kind="bar", stacked=True, ax=ax[5])
ax[5].set_title("Airline Dominance on Routes")
ax[5].set_xlabel("Source City")
ax[5].set_ylabel("Flight Count")
ax[5].tick_params(axis='x', rotation=90)

# Plot - 7 Horizontal Bar Chart Most popular city pairs
popular_pairs = df.groupby(["source_city", "destination_city"])["flight"].count().sort_values(ascending=False).head(10)
popular_pairs.plot(kind="bar", ax=ax[6])
ax[6].set_title("Most Popular City Pairs")
ax[6].set_xlabel("Flight Count")
ax[6].set_ylabel("Source City")


# Plot - 7 Pie Chart Economy vs Business class price
economy_price = df[df["class"] == "Economy"]["price"].mean()
business_price = df[df["class"] == "Business"]["price"].mean()
ax[7].pie([economy_price, business_price], labels=["Economy", "Business"], autopct='%1.1f%%', startangle=90)
ax[7].set_title("Economy vs Business Class Price")


# Layout
plt.tight_layout(pad=4.0)
fig.subplots_adjust(top=0.90, hspace=0.7, wspace=0.3)
fig.suptitle("Employees Data Visualizations", fontsize=20, fontweight="bold")
plt.savefig("Airline Flight Data Visualization.png" , dpi = 300 , bbox_inches = "tight")

plt.show()
