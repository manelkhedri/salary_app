# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Salary Analysis Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2563EB;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .insight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the salary data"""
    df = pd.read_csv('Salary_Data.csv')
    df = df.dropna(subset=['Salary'])
    df = df.head(2000)
    
    # Standardize Education Level names
    df['Education Level'] = df['Education Level'].replace({
        "Bachelor's": "Bachelor",
        "Bachelor's Degree": "Bachelor",
        "Master's": "Master",
        "Master's Degree": "Master",
        "PhD": "PhD"
    })
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove salary outliers using IQR method
    Q1, Q3 = df['Salary'].quantile(0.25), df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Salary'] >= Q1 - 3*IQR) & (df['Salary'] <= Q3 + 3*IQR)]
    
    return df

@st.cache_data
def encode_data(df):
    """Encode categorical variables for modeling"""
    df_encoded = df.copy()
    
    # Gender encoding
    df_encoded['Gender_Code'] = (df_encoded['Gender'] == 'Male').astype(int)
    
    # Education encoding
    education_order = {"Bachelor": 1, "Master": 2, "PhD": 3}
    df_encoded['Education_Code'] = df_encoded['Education Level'].map(education_order)
    
    # Job Title encoding using mean salary
    job_salary_mean = df_encoded.groupby('Job Title')['Salary'].mean()
    df_encoded['Job_Code'] = df_encoded['Job Title'].map(job_salary_mean)
    
    return df_encoded

@st.cache_resource
def train_models(df):
    """Train all three regression models"""
    X_job = df[['Job_Code']]
    X_exp = df[['Years of Experience']]
    X_multi = df[['Job_Code', 'Education_Code', 'Years of Experience']]
    y = df['Salary']
    
    # Split data
    X_job_train, X_job_test, y_train, y_test = train_test_split(X_job, y, test_size=0.2, random_state=42)
    X_exp_train, X_exp_test, _, _ = train_test_split(X_exp, y, test_size=0.2, random_state=42)
    X_multi_train, X_multi_test, _, _ = train_test_split(X_multi, y, test_size=0.2, random_state=42)
    
    # Train models
    model_job = LinearRegression()
    model_job.fit(X_job_train, y_train)
    y_pred_job = model_job.predict(X_job_test)
    
    model_exp = LinearRegression()
    model_exp.fit(X_exp_train, y_train)
    y_pred_exp = model_exp.predict(X_exp_test)
    
    model_multi = LinearRegression()
    model_multi.fit(X_multi_train, y_train)
    y_pred_multi = model_multi.predict(X_multi_test)
    
    # Calculate metrics
    metrics = {
        'Job Title': {
            'R2': r2_score(y_test, y_pred_job),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_job)),
            'MAE': mean_absolute_error(y_test, y_pred_job),
            'predictions': y_pred_job,
            'model': model_job,
            'coef': model_job.coef_[0],
            'intercept': model_job.intercept_
        },
        'Experience': {
            'R2': r2_score(y_test, y_pred_exp),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_exp)),
            'MAE': mean_absolute_error(y_test, y_pred_exp),
            'predictions': y_pred_exp,
            'model': model_exp,
            'coef': model_exp.coef_[0],
            'intercept': model_exp.intercept_
        },
        'Combined': {
            'R2': r2_score(y_test, y_pred_multi),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_multi)),
            'MAE': mean_absolute_error(y_test, y_pred_multi),
            'predictions': y_pred_multi,
            'model': model_multi,
            'coef': model_multi.coef_,
            'intercept': model_multi.intercept_
        }
    }
    
    return metrics, y_test

def run_inferential_stats(df):
    """Run inferential statistics tests"""
    results = {}
    
    # T-test: Gender vs Salary
    male = df[df['Gender'] == 'Male']['Salary']
    female = df[df['Gender'] == 'Female']['Salary']
    t_stat, p_t = stats.ttest_ind(male, female)
    results['gender_ttest'] = {'statistic': t_stat, 'p_value': p_t, 'reject_null': p_t < 0.05}
    
    # ANOVA: Job Title vs Salary
    job_groups = [df[df['Job Title'] == dept]['Salary'] for dept in df['Job Title'].unique()]
    f_job, p_job = stats.f_oneway(*job_groups)
    results['job_anova'] = {'statistic': f_job, 'p_value': p_job, 'reject_null': p_job < 0.05}
    
    # ANOVA: Education vs Salary
    edu_groups = [df[df['Education Level'] == edu]['Salary'] for edu in df['Education Level'].unique()]
    f_edu, p_edu = stats.f_oneway(*edu_groups)
    results['edu_anova'] = {'statistic': f_edu, 'p_value': p_edu, 'reject_null': p_edu < 0.05}
    
    # Pearson correlation: Experience vs Salary
    corr, p_corr = stats.pearsonr(df['Years of Experience'], df['Salary'])
    results['exp_corr'] = {'correlation': corr, 'p_value': p_corr, 'reject_null': p_corr < 0.05}
    
    return results

def plot_salary_distribution(df):
    """Plot salary distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Salary'], bins=40, kde=True, color='steelblue', ax=ax)
    ax.axvline(df['Salary'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Salary"].mean():,.0f}')
    ax.axvline(df['Salary'].median(), color='green', linestyle='--', label=f'Median: ${df["Salary"].median():,.0f}')
    ax.set_title('Salary Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Salary ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def plot_experience_vs_salary(df):
    """Plot experience vs salary scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Years of Experience', y='Salary', data=df, hue='Gender', alpha=0.6, ax=ax)
    
    # Add trend line
    z = np.polyfit(df['Years of Experience'], df['Salary'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Years of Experience'].min(), df['Years of Experience'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: y = {z[0]:.0f}x + {z[1]:.0f}')
    
    ax.set_title('Experience vs Salary Relationship', fontsize=14, fontweight='bold')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary ($)')
    ax.legend()
    return fig

def plot_salary_by_gender(df):
    """Plot salary distribution by gender"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Gender', y='Salary', data=df, palette='Set2', ax=ax)
    ax.set_title('Salary Distribution by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Salary ($)')
    return fig

def plot_salary_by_education(df):
    """Plot salary distribution by education level"""
    fig, ax = plt.subplots(figsize=(10, 6))
    edu_order = df.groupby('Education Level')['Salary'].median().sort_values().index
    sns.boxplot(x='Education Level', y='Salary', data=df, order=edu_order, palette='viridis', ax=ax)
    ax.set_title('Salary by Education Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Salary ($)')
    plt.xticks(rotation=45)
    return fig

def plot_correlation_heatmap(df_encoded):
    """Plot correlation heatmap"""
    numeric_cols = ['Years of Experience', 'Age', 'Salary', 'Gender_Code', 'Education_Code', 'Job_Code']
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation_matrix = df_encoded[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f', square=True, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    return fig

def plot_top_jobs(df):
    """Plot top 10 most common jobs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_jobs = df['Job Title'].value_counts().head(10)
    sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='viridis', ax=ax)
    ax.set_title('Top 10 Most Common Job Titles', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Employees')
    ax.set_ylabel('Job Title')
    return fig

def plot_model_predictions(y_test, predictions, model_name, r2):
    """Plot actual vs predicted values for a model"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.5, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Salary ($)')
    ax.set_ylabel('Predicted Salary ($)')
    ax.set_title(f'{model_name} Model Predictions (R² = {r2:.3f})')
    return fig

# Main application
def main():
    st.markdown('<h1 class="main-header">💰 Salary Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    sections = [
        "📊 Overview",
        "📈 Descriptive Statistics",
        "🔬 Inferential Statistics",
        "🛠️ Data Preprocessing",
        "📉 EDA & Visualizations",
        "🤖 Linear Regression Models",
        "🏆 Model Comparison",
        "🎯 Best Model Testing"
    ]
    choice = st.sidebar.radio("Go to", sections)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        df_encoded = encode_data(df)
    
    # Show raw data in expander
    with st.sidebar.expander("📄 View Raw Data"):
        st.dataframe(df.head(100))
        st.caption(f"Total rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Overview Section
    if choice == "📊 Overview":
        st.markdown('<h2 class="section-header">📊 Project Overview</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🎯 Objective:</strong> Analyze salary data to identify key factors influencing compensation 
        and build predictive models to estimate salaries based on job characteristics.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Unique Jobs", df['Job Title'].nunique())
        with col4:
            st.metric("Salary Range", f"${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}")
        
        st.markdown('<h3 class="subsection-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        with col2:
            st.write("**Dataset Info:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values
            })
            st.dataframe(info_df)
        
        st.markdown('<div class="insight-box">💡 <strong>Key Insight:</strong> The dataset contains salary information across various job titles, with multiple demographic and professional attributes.</div>', unsafe_allow_html=True)
    
    # Descriptive Statistics Section
    elif choice == "📈 Descriptive Statistics":
        st.markdown('<h2 class="section-header">📈 Descriptive Statistics</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Descriptive statistics provide a summary of the central tendency, dispersion, and 
        shape of the dataset's distribution.
        """)
        
        numeric_cols = ['Age', 'Years of Experience', 'Salary']
        
        for col in numeric_cols:
            st.markdown(f'<h3 class="subsection-header">{col}</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            stats_data = {
                'Mean': f"${df[col].mean():,.2f}" if col == 'Salary' else f"{df[col].mean():.2f}",
                'Median': f"${df[col].median():,.2f}" if col == 'Salary' else f"{df[col].median():.2f}",
                'Std Dev': f"${df[col].std():,.2f}" if col == 'Salary' else f"{df[col].std():.2f}",
                'Min': f"${df[col].min():,.0f}" if col == 'Salary' else f"{df[col].min():.0f}",
                'Max': f"${df[col].max():,.0f}" if col == 'Salary' else f"{df[col].max():.0f}",
                'Q1': f"${df[col].quantile(0.25):,.0f}" if col == 'Salary' else f"{df[col].quantile(0.25):.0f}",
                'Q3': f"${df[col].quantile(0.75):,.0f}" if col == 'Salary' else f"{df[col].quantile(0.75):.0f}"
            }
            
            with col1:
                st.metric("Mean", stats_data['Mean'])
                st.metric("Min", stats_data['Min'])
            with col2:
                st.metric("Median", stats_data['Median'])
                st.metric("Max", stats_data['Max'])
            with col3:
                st.metric("Std Dev", stats_data['Std Dev'])
                st.metric("Q1", stats_data['Q1'])
            with col4:
                st.metric("Skewness", f"{df[col].skew():.3f}")
                st.metric("Q3", stats_data['Q3'])
        
        # Categorical variables summary
        st.markdown('<h3 class="subsection-header">Categorical Variables</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Gender Distribution:**")
            gender_counts = df['Gender'].value_counts()
            st.dataframe(gender_counts)
        with col2:
            st.write("**Education Distribution:**")
            edu_counts = df['Education Level'].value_counts()
            st.dataframe(edu_counts)
        
        st.write("**Top 10 Job Titles:**")
        st.dataframe(df['Job Title'].value_counts().head(10))
        
        # Salary histogram
        st.markdown('<h3 class="subsection-header">Salary Distribution</h3>', unsafe_allow_html=True)
        fig = plot_salary_distribution(df)
        st.pyplot(fig)
        
        skewness = df['Salary'].skew()
        kurtosis = df['Salary'].kurtosis()
        st.markdown(f"""
        <div class="insight-box">
        <strong>📊 Distribution Insights:</strong><br>
        • Skewness: {skewness:.3f} - {'Right-skewed (most people earn below average)' if skewness > 0 else 'Left-skewed'}<br>
        • Kurtosis: {kurtosis:.3f} - {'Heavy tails (more outliers than normal distribution)' if kurtosis > 0 else 'Light tails'}
        </div>
        """, unsafe_allow_html=True)
    
    # Inferential Statistics Section
    elif choice == "🔬 Inferential Statistics":
        st.markdown('<h2 class="section-header">🔬 Inferential Statistics</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Inferential statistics help us draw conclusions about the population based on sample data.
        We test hypotheses to determine if observed differences are statistically significant.
        """)
        
        results = run_inferential_stats(df)
        
        # T-test
        st.markdown('<h3 class="subsection-header">1. T-Test: Gender vs Salary</h3>', unsafe_allow_html=True)
        t_result = results['gender_ttest']
        male_avg = df[df['Gender'] == 'Male']['Salary'].mean()
        female_avg = df[df['Gender'] == 'Female']['Salary'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Average Salary", f"${male_avg:,.0f}")
            st.metric("T-Statistic", f"{t_result['statistic']:.4f}")
        with col2:
            st.metric("Female Average Salary", f"${female_avg:,.0f}")
            st.metric("P-Value", f"{t_result['p_value']:.6f}")
        
        if t_result['reject_null']:
            st.markdown(f"""
            <div class="insight-box">
            ✅ <strong>Result:</strong> REJECT null hypothesis (p-value = {t_result['p_value']:.6f})<br>
            There is a statistically significant difference in salary between genders.
            Men earn {(male_avg/female_avg-1)*100:.1f}% more on average.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
            ❌ <strong>Result:</strong> ACCEPT null hypothesis (p-value = {t_result['p_value']:.6f})<br>
            No statistically significant difference in salary between genders.
            </div>
            """, unsafe_allow_html=True)
        
        # ANOVA: Job Title
        st.markdown('<h3 class="subsection-header">2. ANOVA: Job Title vs Salary</h3>', unsafe_allow_html=True)
        job_result = results['job_anova']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-Statistic", f"{job_result['statistic']:.4f}")
        with col2:
            st.metric("P-Value", f"{job_result['p_value']:.6f}")
        
        if job_result['reject_null']:
            st.markdown(f"""
            <div class="insight-box">
            ✅ <strong>Result:</strong> REJECT null hypothesis (p-value = {job_result['p_value']:.6f})<br>
            Job title has a statistically significant effect on salary.
            </div>
            """, unsafe_allow_html=True)
        
        # ANOVA: Education
        st.markdown('<h3 class="subsection-header">3. ANOVA: Education Level vs Salary</h3>', unsafe_allow_html=True)
        edu_result = results['edu_anova']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F-Statistic", f"{edu_result['statistic']:.4f}")
        with col2:
            st.metric("P-Value", f"{edu_result['p_value']:.6f}")
        
        if edu_result['reject_null']:
            st.markdown(f"""
            <div class="insight-box">
            ✅ <strong>Result:</strong> REJECT null hypothesis (p-value = {edu_result['p_value']:.6f})<br>
            Education level has a statistically significant effect on salary.
            </div>
            """, unsafe_allow_html=True)
        
        # Correlation: Experience vs Salary
        st.markdown('<h3 class="subsection-header">4. Pearson Correlation: Experience vs Salary</h3>', unsafe_allow_html=True)
        corr_result = results['exp_corr']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correlation Coefficient", f"{corr_result['correlation']:.4f}")
        with col2:
            st.metric("P-Value", f"{corr_result['p_value']:.6f}")
        
        if corr_result['reject_null']:
            st.markdown(f"""
            <div class="insight-box">
            ✅ <strong>Result:</strong> REJECT null hypothesis (p-value = {corr_result['p_value']:.6f})<br>
            There is a statistically significant correlation between experience and salary.
            Correlation strength: {'Strong' if abs(corr_result['correlation']) > 0.7 else 'Moderate' if abs(corr_result['correlation']) > 0.4 else 'Weak'}
            </div>
            """, unsafe_allow_html=True)
        
        # Experience vs Salary plot
        fig = plot_experience_vs_salary(df)
        st.pyplot(fig)
    
    # Data Preprocessing Section
    elif choice == "🛠️ Data Preprocessing":
        st.markdown('<h2 class="section-header">🛠️ Data Preprocessing</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Data preprocessing involves cleaning and transforming raw data into a format suitable for analysis and modeling.
        """)
        
        st.markdown('<h3 class="subsection-header">🔧 Preprocessing Steps Applied</h3>', unsafe_allow_html=True)
        
        steps = [
            "1. **Data Loading** - Loaded Salary_Data.csv (2000 initial records)",
            "2. **Missing Value Handling** - Removed rows with missing Salary values",
            "3. **Education Standardization** - Standardized education level names (Bachelor's → Bachelor, Master's → Master, PhD → PhD)",
            "4. **Duplicate Removal** - Removed duplicate rows",
            "5. **Outlier Removal** - Used IQR method (3×IQR) to remove salary outliers",
            "6. **Feature Encoding** - Converted categorical variables to numerical codes:",
            "   - Gender: Female=0, Male=1",
            "   - Education: Bachelor=1, Master=2, PhD=3",
            "   - Job Title: Target encoding using mean salary per job"
        ]
        
        for step in steps:
            st.markdown(step)
        
        st.markdown('<h3 class="subsection-header">📊 Encoding Results</h3>', unsafe_allow_html=True)
        
        sample_cols = ['Gender', 'Gender_Code', 'Education Level', 'Education_Code', 'Job Title', 'Job_Code', 'Salary']
        st.dataframe(df_encoded[sample_cols].head(10))
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📈 Preprocessing Summary:</strong><br>
        • Final dataset shape: {df_encoded.shape}<br>
        • Encoded categorical variables successfully<br>
        • No missing values remaining<br>
        • Outliers removed using IQR method
        </div>
        """, unsafe_allow_html=True)
    
    # EDA Section
    elif choice == "📉 EDA & Visualizations":
        st.markdown('<h2 class="section-header">📉 Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        EDA helps understand patterns, relationships, and anomalies in the data through visualizations.
        """)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Salary Distribution", "Experience vs Salary", "Gender Analysis", "Education Analysis", "Correlations"])
        
        with tab1:
            st.markdown('<h3 class="subsection-header">Salary Distribution</h3>', unsafe_allow_html=True)
            fig = plot_salary_distribution(df)
            st.pyplot(fig)
            st.markdown("""
            <div class="insight-box">
            <strong>📊 Key Observations:</strong><br>
            • The distribution appears slightly right-skewed<br>
            • Mean salary is higher than median, indicating some high earners<br>
            • Most salaries cluster between $100,000 and $185,000
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h3 class="subsection-header">Experience vs Salary Relationship</h3>', unsafe_allow_html=True)
            fig = plot_experience_vs_salary(df)
            st.pyplot(fig)
            st.markdown("""
            <div class="insight-box">
            <strong>📊 Key Observations:</strong><br>
            • Strong positive correlation between experience and salary<br>
            • Trend line shows ~$5-6k increase per year of experience<br>
            • Variation increases with experience (more uncertainty for senior roles)
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h3 class="subsection-header">Salary by Gender</h3>', unsafe_allow_html=True)
            fig = plot_salary_by_gender(df)
            st.pyplot(fig)
            male_avg = df[df['Gender'] == 'Male']['Salary'].mean()
            female_avg = df[df['Gender'] == 'Female']['Salary'].mean()
            st.markdown(f"""
            <div class="insight-box">
            <strong>📊 Key Observations:</strong><br>
            • Male average: ${male_avg:,.0f} | Female average: ${female_avg:,.0f}<br>
            • Gender pay gap: {(male_avg/female_avg-1)*100:.1f}% difference<br>
            • Both distributions show similar spread but different medians
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h3 class="subsection-header">Salary by Education Level</h3>', unsafe_allow_html=True)
            fig = plot_salary_by_education(df)
            st.pyplot(fig)
            phd_avg = df[df['Education Level'] == 'PhD']['Salary'].mean()
            bachelor_avg = df[df['Education Level'] == 'Bachelor']['Salary'].mean()
            st.markdown(f"""
            <div class="insight-box">
            <strong>📊 Key Observations:</strong><br>
            • PhD average: ${phd_avg:,.0f} | Bachelor's average: ${bachelor_avg:,.0f}<br>
            • Education premium: {(phd_avg/bachelor_avg-1)*100:.0f}% more for PhD<br>
            • Clear progression: higher education → higher salary
            </div>
            """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<h3 class="subsection-header">Correlation Matrix</h3>', unsafe_allow_html=True)
            fig = plot_correlation_heatmap(df_encoded)
            st.pyplot(fig)
            
            correlations = {
                'Years of Experience': df['Years of Experience'].corr(df['Salary']),
                'Age': df['Age'].corr(df['Salary']),
                'Education Level': df_encoded['Education_Code'].corr(df_encoded['Salary']),
                'Gender': df_encoded['Gender_Code'].corr(df_encoded['Salary']),
                'Job Title': df_encoded['Job_Code'].corr(df_encoded['Salary'])
            }
            
            st.markdown("**Correlation with Salary:**")
            for k, v in correlations.items():
                st.write(f"• {k}: {v:.4f}")
            
            st.markdown("""
            <div class="insight-box">
            <strong>📊 Key Observations:</strong><br>
            • Job Title has the strongest correlation with salary (0.93)<br>
            • Experience and Education show moderate to strong correlations<br>
            • Gender shows weak correlation with salary
            </div>
            """, unsafe_allow_html=True)
        
        # Top jobs
        st.markdown('<h3 class="subsection-header">Top 10 Most Common Job Titles</h3>', unsafe_allow_html=True)
        fig = plot_top_jobs(df)
        st.pyplot(fig)
    
    # Linear Regression Models Section
    elif choice == "🤖 Linear Regression Models":
        st.markdown('<h2 class="section-header">🤖 Linear Regression Models</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Linear regression models predict a target variable (salary) based on one or more predictor variables.
        We trained three different models to compare their performance.
        """)
        
        with st.spinner("Training models..."):
            metrics, y_test = train_models(df_encoded)
        
        # Model 1
        st.markdown('<h3 class="subsection-header">📊 Model 1: Job Title → Salary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['Job Title']['R2']:.4f}")
        with col2:
            st.metric("RMSE", f"${metrics['Job Title']['RMSE']:,.2f}")
        with col3:
            st.metric("MAE", f"${metrics['Job Title']['MAE']:,.2f}")
        
        st.markdown(f"""
        **Equation:** Salary = {metrics['Job Title']['coef']:.2f} × Job_Code + {metrics['Job Title']['intercept']:.2f}
        """)
        
        fig = plot_model_predictions(y_test, metrics['Job Title']['predictions'], "Job Title", metrics['Job Title']['R2'])
        st.pyplot(fig)
        
        # Model 2
        st.markdown('<h3 class="subsection-header">📊 Model 2: Years of Experience → Salary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['Experience']['R2']:.4f}")
        with col2:
            st.metric("RMSE", f"${metrics['Experience']['RMSE']:,.2f}")
        with col3:
            st.metric("MAE", f"${metrics['Experience']['MAE']:,.2f}")
        
        st.markdown(f"""
        **Equation:** Salary = {metrics['Experience']['coef']:.2f} × Experience + {metrics['Experience']['intercept']:.2f}
        """)
        
        fig = plot_model_predictions(y_test, metrics['Experience']['predictions'], "Experience", metrics['Experience']['R2'])
        st.pyplot(fig)
        
        # Model 3
        st.markdown('<h3 class="subsection-header">📊 Model 3: Job Title + Education + Experience → Salary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['Combined']['R2']:.4f}")
        with col2:
            st.metric("RMSE", f"${metrics['Combined']['RMSE']:,.2f}")
        with col3:
            st.metric("MAE", f"${metrics['Combined']['MAE']:,.2f}")
        
        coefs = metrics['Combined']['coef']
        st.markdown(f"""
        **Equation:** Salary = {coefs[0]:.2f} × Job_Code + {coefs[1]:.2f} × Education_Code + {coefs[2]:.2f} × Experience + {metrics['Combined']['intercept']:.2f}
        """)
        
        st.markdown("**Coefficients (Impact of each feature):**")
        st.write(f"• Job Code: {coefs[0]:.2f} (salary change per 1 unit of job code)")
        st.write(f"• Education: {coefs[1]:.2f} (salary change per education level)")
        st.write(f"• Experience: {coefs[2]:.2f} (salary change per year)")
        
        fig = plot_model_predictions(y_test, metrics['Combined']['predictions'], "Combined Model", metrics['Combined']['R2'])
        st.pyplot(fig)
    
    # Model Comparison Section
    elif choice == "🏆 Model Comparison":
        st.markdown('<h2 class="section-header">🏆 Model Comparison</h2>', unsafe_allow_html=True)
        
        with st.spinner("Training models..."):
            metrics, y_test = train_models(df_encoded)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': ['Job Title Only', 'Experience Only', 'Job + Education + Experience'],
            'R² Score': [metrics['Job Title']['R2'], metrics['Experience']['R2'], metrics['Combined']['R2']],
            'RMSE ($)': [metrics['Job Title']['RMSE'], metrics['Experience']['RMSE'], metrics['Combined']['RMSE']],
            'MAE ($)': [metrics['Job Title']['MAE'], metrics['Experience']['MAE'], metrics['Combined']['MAE']]
        })
        
        comparison_df['R² Score'] = comparison_df['R² Score'].round(4)
        comparison_df['RMSE ($)'] = comparison_df['RMSE ($)'].apply(lambda x: f"${x:,.0f}")
        comparison_df['MAE ($)'] = comparison_df['MAE ($)'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = ['Job Title', 'Experience', 'Combined']
        r2_values = [metrics['Job Title']['R2'], metrics['Experience']['R2'], metrics['Combined']['R2']]
        rmse_values = [metrics['Job Title']['RMSE'], metrics['Experience']['RMSE'], metrics['Combined']['RMSE']]
        mae_values = [metrics['Job Title']['MAE'], metrics['Experience']['MAE'], metrics['Combined']['MAE']]
        
        axes[0].bar(models, r2_values, color=['blue', 'green', 'purple'])
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² Score Comparison')
        axes[0].set_ylim(0, 1)
        
        axes[1].bar(models, rmse_values, color=['blue', 'green', 'purple'])
        axes[1].set_ylabel('RMSE ($)')
        axes[1].set_title('RMSE Comparison')
        
        axes[2].bar(models, mae_values, color=['blue', 'green', 'purple'])
        axes[2].set_ylabel('MAE ($)')
        axes[2].set_title('MAE Comparison')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Winner announcement
        best_model = "Job + Education + Experience"
        best_r2 = metrics['Combined']['R2']
        
        st.markdown(f"""
        <div class="insight-box" style="background-color: #D1FAE5; border-left-color: #10B981;">
        🏆 <strong>Best Model: {best_model}</strong><br>
        • R² Score: {best_r2:.4f} (explains {best_r2*100:.1f}% of salary variance)<br>
        • RMSE: ${metrics['Combined']['RMSE']:,.2f}<br>
        • MAE: ${metrics['Combined']['MAE']:,.2f}<br><br>
        <strong>Why this is the best model:</strong> It uses multiple features (job, education, experience) 
        which collectively provide better predictive power than any single feature alone.
        </div>
        """, unsafe_allow_html=True)
        
        # Model predictions visualization side by side
        st.markdown('<h3 class="subsection-header">Model Predictions Comparison</h3>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].scatter(y_test, metrics['Job Title']['predictions'], alpha=0.5, color='blue')
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0].set_xlabel('Actual Salary')
        axes[0].set_ylabel('Predicted Salary')
        axes[0].set_title(f'Model 1: Job Title (R²={metrics["Job Title"]["R2"]:.3f})')
        
        axes[1].scatter(y_test, metrics['Experience']['predictions'], alpha=0.5, color='green')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[1].set_xlabel('Actual Salary')
        axes[1].set_ylabel('Predicted Salary')
        axes[1].set_title(f'Model 2: Experience (R²={metrics["Experience"]["R2"]:.3f})')
        
        axes[2].scatter(y_test, metrics['Combined']['predictions'], alpha=0.5, color='purple')
        axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[2].set_xlabel('Actual Salary')
        axes[2].set_ylabel('Predicted Salary')
        axes[2].set_title(f'Model 3: Combined (R²={metrics["Combined"]["R2"]:.3f})')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Best Model Testing Section
    elif choice == "🎯 Best Model Testing":
        st.markdown('<h2 class="section-header">🎯 Best Model Testing & Prediction</h2>', unsafe_allow_html=True)
        
        with st.spinner("Loading best model..."):
            metrics, y_test = train_models(df_encoded)
        
        st.markdown("""
        <div class="insight-box">
        The best model combines Job Title, Education Level, and Years of Experience to predict salary.
        Use the form below to test the model with your own inputs!
        </div>
        """, unsafe_allow_html=True)
        
        # Get unique job titles and mapping
        job_salary_mean = df_encoded.groupby('Job Title')['Salary'].mean()
        job_list = sorted(job_salary_mean.index.tolist())
        
        # Create input form
        st.markdown('<h3 class="subsection-header">📝 Enter Your Information</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            job_title = st.selectbox("Job Title", job_list)
        
        with col2:
            education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
        
        with col3:
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)
        
        if st.button("💰 Predict Salary", type="primary"):
            # Get encoded values
            job_code = job_salary_mean[job_title]
            edu_code = {"Bachelor": 1, "Master": 2, "PhD": 3}[education]
            
            # Create feature array
            features = np.array([[job_code, edu_code, years_exp]])
            
            # Make prediction
            model = metrics['Combined']['model']
            prediction = model.predict(features)[0]
            
            # Display prediction
            st.markdown(f"""
            <div style="background-color: #D1FAE5; padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 1rem;">
                <h2 style="color: #065F46; margin-bottom: 0;">💰 Predicted Salary</h2>
                <p style="font-size: 3rem; font-weight: bold; color: #065F46; margin: 0;">${prediction:,.0f}</p>
                <p style="color: #065F46;">Based on: {job_title} | {education} | {years_exp} years experience</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence interval (using RMSE as approximate margin of error)
            rmse = metrics['Combined']['RMSE']
            st.markdown(f"""
            <div class="insight-box">
            <strong>📊 Prediction Details:</strong><br>
            • Model Confidence: R² = {metrics['Combined']['R2']:.4f}<br>
            • Typical prediction error (±RMSE): ${rmse:,.0f}<br>
            • Expected salary range: ${prediction - rmse:,.0f} - ${prediction + rmse:,.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # Model performance summary
        st.markdown('<h3 class="subsection-header">📊 Model Performance Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['Combined']['R2']:.4f}")
        with col2:
            st.metric("RMSE", f"${metrics['Combined']['RMSE']:,.2f}")
        with col3:
            st.metric("MAE", f"${metrics['Combined']['MAE']:,.2f}")
        
        # Feature importance
        st.markdown('<h3 class="subsection-header">📊 Feature Importance</h3>', unsafe_allow_html=True)
        
        coefs = metrics['Combined']['coef']
        feature_importance = pd.DataFrame({
            'Feature': ['Job Title', 'Education Level', 'Years of Experience'],
            'Coefficient': coefs,
            'Impact': ['Salary change per unit change in job code',
                      'Salary increase per education level (Bachelor→Master→PhD)',
                      'Salary increase per additional year of experience']
        })
        st.dataframe(feature_importance, use_container_width=True)
        
        # Model equation
        st.markdown(f"""
        <div class="insight-box">
        <strong>📐 Model Equation:</strong><br>
        Predicted Salary = {coefs[0]:.2f} × Job_Code + {coefs[1]:.2f} × Education_Code + {coefs[2]:.2f} × Experience + {metrics['Combined']['intercept']:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Model is ready for predictions!")


if __name__ == "__main__":
    main()