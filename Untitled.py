#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


data = pd.read_excel('data.xlsx')
data


# In[11]:


data.isna()


# In[3]:


data.isna().sum()


# In[4]:


data.head()


# In[5]:


import statistics as st


# In[6]:


# Calculate mean, median, standard deviation, skewness, and kurtosis for numeric columns
mean_values = data.mean()
median_values = data.median()
std_deviation = data.std()
skewness = data.skew()
kurtosis = data.kurtosis()


# In[7]:


#3 end
# Display the calculated statistics
print("Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)

print("\nStandard Deviation:")
print(std_deviation)

print("\nSkewness:")
print(skewness)

print("\nKurtosis:")
print(kurtosis)


# In[8]:


#4
# Display the column names in your DataFrame
print(data.columns)


# Define segmentation criteria (e.g., by year)
years_to_segment = [2015, 2016, 2017]

# Create a new column to store the segmentation result
data['Segment'] = None

# Use conditional statements to categorize the data
for year in years_to_segment:
    data.loc[data['year'] == year, 'Segment'] = f'year {year}'

# Display the segmented data
segmented_data = data[['year', 'Segment']]
print(segmented_data.head())


# In[9]:


#ph2-1-bar
import matplotlib.pyplot as plt

# Count the number of cars in each segment
segment_counts = segmented_data['Segment'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
segment_counts.plot(kind='bar')
plt.title('Segmentation of Used Toyota Cars')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[19]:


#ph-2-1-line
# Sample data (x-axis values and corresponding y-axis values)
years = [2010, 2011, 2012, 2013, 2014, 2015]
car_sales = [120, 150, 200, 180, 220, 250]

# Create a line chart
plt.figure(figsize=(8, 6))
plt.plot(years, car_sales, marker='o', linestyle='-', color='b', label='Car Sales')
plt.title('Toyota Car Sales Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Cars Sold')
plt.legend()
plt.grid(True)

# Display the line chart
plt.show()



# In[20]:


#ph-2-2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your dataset file path)


# Define your feature columns (independent variables) and target column (car price)
X = data[['year', 'mileage', 'tax', 'mpg']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
# Plot the predicted vs. actual prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs. Predicted Car Prices")
plt.show()


# In[21]:


#ph-2-3
from scipy.stats import norm  # Import the norm function from scipy.stats

# Replace 'data' with your DataFrame and 'column_name' with the column you want to visualize
column_name = 'mileage'

# Create a histogram
plt.figure(figsize=(8, 6))
plt.hist(data[column_name], bins=20, density=True, alpha=0.6, color='b')
plt.title(f'Normal Distribution of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Frequency')

# Add a normal distribution curve for comparison (if the data is approximately normally distributed)
mean = data[column_name].mean()
std = data[column_name].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Sample car data (replace with your own data)
car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"]
car_prices = [25000, 27000, 23000, 22000, 26000]  # Prices in USD
fuel_efficiency = [30, 32, 28, 29, 31]  # MPG (miles per gallon)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(car_prices, fuel_efficiency, label="Cars", color="blue", marker="o")

# Add labels and title
plt.xlabel("Price (USD)")
plt.ylabel("Fuel Efficiency (MPG)")
plt.title("Car Prices vs. Fuel Efficiency")

# Add data labels
for i, brand in enumerate(car_brands):
    plt.annotate(brand, (car_prices[i], fuel_efficiency[i]), textcoords="offset points", xytext=(0, 10), ha="center")

# Display the legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()



# In[12]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[13]:


img = mpg.imread("toyota_logo.png")
plt.figure(figsize=(6, 6))  
plt.imshow(img)
plt.axis("off")  
# Add a title (optional)
plt.title("Toyota Logo")

# Show the plot
plt.show()


# In[ ]:




