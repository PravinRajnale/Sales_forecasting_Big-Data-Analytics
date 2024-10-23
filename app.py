import os
import base64
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def generate_insights(evaluation, df):
    total_sales = df['Total Sales'].sum()
    max_product = df.groupby('Product Name')['Total Sales'].sum().idxmax()
    max_product_sales = df.groupby('Product Name')['Total Sales'].sum().max()

    prompt = f"""
    You are analyzing the sales dataset provided in the file. The dataset includes information on product names, sales amounts, and relevant details across different time periods.
    
    1. **Data Analysis:** 
        - Study and analyze the total sales amount from the dataset, breaking it down by individual product names, sales trends, and any anomalies.
        - Identify the top-performing products by sales volume, including the product with the highest sales, {max_product}, which had sales of {max_product_sales}.
        - Highlight any seasonal trends or patterns observed within the sales data.
        
    2. **Model Evaluation:** 
        - The evaluation results for the forecasting models are as follows: {evaluation}. Analyze the performance of these models based on their accuracy in predicting total sales and individual product trends.
        
    3. **Insights & Recommendations:**
        - Based on the analysis, provide **actionable insights** on how to optimize sales for the top-performing products and underperforming ones.
        - Offer **specific recommendations** supported by numerical data (e.g., percentages, sales amounts, product names).
        - Suggest strategies to improve forecasting accuracy for the models evaluated, targeting specific product categories, regions, or time frames where the models may have underperformed.
        - Mention areas of improvement for each model with regard to prediction accuracy, sales patterns, or other metrics.

    4. **Additional Areas to Focus:**
        - Provide strategic insights to improve overall sales based on product demand, market segmentation, or pricing adjustments.
        - Identify growth opportunities for products with high potential but currently lower sales, and suggest actionable strategies (e.g., marketing, distribution, or bundling) backed by sales data.
"""
    return prompt

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('dashboard', filename=file.filename))
        else:
            flash('Allowed file types are .xlsx, .xls')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/dashboard/<filename>')
def dashboard(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error loading data: {e}')
        return redirect(url_for('index'))

    data_info = df.info(buf=io.StringIO())
    data_description = df.describe().to_html(classes="table table-striped", border=0)

    try:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    except Exception as e:
        flash(f'Data preprocessing error: {e}')
        return redirect(request.url)

    df.dropna(inplace=True)
    df['Total Sales'] = df['Sales'] * df['Quantity']

    X = df[['Quantity', 'Discount', 'Profit']]
    y = df['Total Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    # HistGradientBoostingRegressor
    hgb_model = HistGradientBoostingRegressor(random_state=42)
    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)
    mse_hgb = mean_squared_error(y_test, y_pred_hgb)
    r2_hgb = r2_score(y_test, y_pred_hgb)

    # MLP Regressor
    mlp_model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42))
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)

    evaluation = {
        'Gradient Boosting Regressor': {'MSE': round(mse_gb, 2), 'R2 Score': round(r2_gb, 2)},
        'HistGradientBoostingRegressor': {'MSE': round(mse_hgb, 2), 'R2 Score': round(r2_hgb, 2)},
        'MLP Regressor': {'MSE': round(mse_mlp, 2), 'R2 Score': round(r2_mlp, 2)}
    }

    insights = generate_insights(evaluation, df)

    # Scatter plot for Gradient Boosting Regressor
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.6, color='blue', ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_title('Actual vs Predicted Sales (Gradient Boosting Regressor)')
    ax1.set_xlabel('Actual Sales')
    ax1.set_ylabel('Predicted Sales')
    fig1.tight_layout()
    plot1 = generate_plot(fig1)

    # Scatter plot for HistGradientBoostingRegressor
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_hgb, alpha=0.6, color='green', ax=ax2)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title('Actual vs Predicted Sales (HistGradientBoostingRegressor)')
    ax2.set_xlabel('Actual Sales')
    ax2.set_ylabel('Predicted Sales')
    fig2.tight_layout()
    plot2 = generate_plot(fig2)

    # Scatter plot for MLP Regressor
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_mlp, alpha=0.6, color='orange', ax=ax3)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_title('Actual vs Predicted Sales (MLP Regressor)')
    ax3.set_xlabel('Actual Sales')
    ax3.set_ylabel('Predicted Sales')
    fig3.tight_layout()
    plot3 = generate_plot(fig3)

    # Distribution of Total Sales
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Total Sales'], bins=30, kde=True, color='purple', ax=ax4)
    ax4.set_title('Distribution of Total Sales')
    ax4.set_xlabel('Total Sales')
    ax4.set_ylabel('Frequency')
    fig4.tight_layout()
    plot4 = generate_plot(fig4)

    # Correlation Heatmap
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, ax=ax5)
    ax5.set_title('Correlation Heatmap')
    fig5.tight_layout()
    plot5 = generate_plot(fig5)

    # Feature Importance (XGBoost)
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    importances = xgb_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=feature_names[indices], ax=ax6)
    ax6.set_title('Feature Importance (XGBoost)')
    ax6.set_xlabel('Importance')
    fig6.tight_layout()
    plot6 = generate_plot(fig6)

    # Pie chart for sales by product
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sales_by_product = df.groupby('Product Name')['Total Sales'].sum()
    sales_by_product.plot.pie(autopct='%1.1f%%', startangle=90, cmap='coolwarm', ax=ax7)
    ax7.set_ylabel('')
    ax7.set_title('Sales Distribution by Product')
    fig7.tight_layout()
    plot7 = generate_plot(fig7)

    return render_template('dashboard.html', evaluation=evaluation, insights=insights, data_info=data_info, 
                           data_description=data_description, plot1=plot1, plot2=plot2, plot3=plot3, 
                           plot4=plot4, plot5=plot5, plot6=plot6, plot7=plot7)

if __name__ == '__main__':
    app.run(debug=True)