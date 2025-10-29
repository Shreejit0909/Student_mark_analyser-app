"""
Advanced Student Performance Analytics Dashboard Pro
Multi-sheet analysis with stunning visualizations
Run: streamlit run app.py
"""

import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="Student Analytics Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
    /* Main Theme */
    .main-header {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 25px 0 10px 0;
        letter-spacing: -1px;
    }
    .sub-header {
        text-align: center;
        color: #b3b8c3; /* improved contrast on dark bg */
        margin-bottom: 40px;
        font-size: 18px;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Info Boxes */
    .insight-box {
        background: linear-gradient(135deg, #e0f7ff 0%, #f0f9ff 100%);
        padding: 25px;
        border-left: 6px solid #1f77b4;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        color: #0f172a; /* dark text for contrast */
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #fffbf0 100%);
        padding: 20px;
        border-left: 6px solid #ffc107;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        padding: 20px;
        border-left: 6px solid #28a745;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    .danger-box {
        background: linear-gradient(135deg, #ffebee 0%, #fef5f5 100%);
        padding: 20px;
        border-left: 6px solid #dc3545;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        color: #0f172a;
    }
    .insight-box h1, .insight-box h2, .insight-box h3, .insight-box h4, .insight-box p, .insight-box li,
    .warning-box h1, .warning-box h2, .warning-box h3, .warning-box h4, .warning-box p, .warning-box li,
    .success-box h1, .success-box h2, .success-box h3, .success-box h4, .success-box p, .success-box li,
    .danger-box h1, .danger-box h2, .danger-box h3, .danger-box h4, .danger-box p, .danger-box li {
        color: #0f172a !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 28px;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Stat Cards */
    .stat-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border: 2px solid #f0f0f0;
        transition: all 0.3s ease;
        color: #1f2937; /* dark text on light card */
    }
    .stat-card:hover {
        border-color: #667eea;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    .stat-number {
        font-size: 36px;
        font-weight: 800;
        color: #667eea;
        margin: 10px 0;
    }
    .stat-label {
        font-size: 14px;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sheet Badge */
    .sheet-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px;
        font-size: 14px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f3f4f6; /* softer light tile */
        border-radius: 10px;
        padding: 0 20px;
        font-weight: 600;
        color: #1f2937 !important; /* ensure dark text on light tab */
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important; /* keep white text on selected */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e7e9ee;
        color: #111827 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def read_excel_with_multirow_headers(file_bytes, filename, sheet_name=0):
    """Read Excel with complex multi-row headers"""
    try:
        bio = io.BytesIO(file_bytes)
        
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(bio)
            return df, None
        
        excel_file = pd.ExcelFile(bio)
        sheet_names = excel_file.sheet_names
        
        df_raw = pd.read_excel(bio, sheet_name=sheet_name, header=None, engine='openpyxl')
        
        data_start_row = 0
        header_rows = []
        
        for idx in range(min(10, len(df_raw))):
            row_str = ' '.join(df_raw.iloc[idx].astype(str).str.lower())
            
            if any(keyword in row_str for keyword in ['subject', 'marks', 'ut', 're-u']):
                header_rows.append(idx)
            
            if 'student name' in row_str or 'student prn' in row_str:
                data_start_row = idx
                if idx not in header_rows:
                    header_rows.append(idx)
        
        if header_rows:
            bio.seek(0)
            if len(header_rows) > 1:
                df = pd.read_excel(bio, sheet_name=sheet_name, header=header_rows, engine='openpyxl')
                
                new_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        parts = []
                        for part in col:
                            part_str = str(part).strip()
                            if (part_str.lower() not in ['nan', 'unnamed'] and 
                                not part_str.startswith('Unnamed') and
                                part_str != '' and
                                not part_str.lower().startswith('marks (')):
                                parts.append(part_str)
                        
                        combined = ' '.join(parts) if parts else 'Unnamed'
                        new_cols.append(combined)
                    else:
                        new_cols.append(str(col).strip())
                
                df.columns = new_cols
            else:
                df = pd.read_excel(bio, sheet_name=sheet_name, header=header_rows[0], engine='openpyxl')
        else:
            bio.seek(0)
            df = pd.read_excel(bio, sheet_name=sheet_name, engine='openpyxl')
        
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.columns = [str(c).strip() for c in df.columns]
        df = df[~df.iloc[:, 0].astype(str).str.lower().str.contains('sr.no|sr no|serial', na=False)]
        df = df.reset_index(drop=True)
        
        return df, sheet_names
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None


def detect_columns(df):
    """Intelligent column detection"""
    col_map = {
        'sr_no': None,
        'prn': None,
        'name': None,
        'division': None,
        'marks_cols': [],
        'subjects': {}
    }
    
    cols_lower = {col: col.lower().strip() for col in df.columns}
    
    for col, col_l in cols_lower.items():
        if any(term in col_l for term in ['sr.no', 'sr no', 'srno', 'sr_no', 'serial', 's.no', 'sno', 'sr']):
            col_map['sr_no'] = col
            break
    
    for col, col_l in cols_lower.items():
        if 'prn' in col_l:
            col_map['prn'] = col
            break
    
    for col, col_l in cols_lower.items():
        if 'student name' in col_l or col_l == 'name':
            col_map['name'] = col
            break

    # Try to detect division/class/section column
    for col, col_l in cols_lower.items():
        if any(term in col_l for term in ['division', 'div', 'section', 'class']):
            col_map['division'] = col
            break
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(term in col_lower for term in ['ut 1', 'ut 2', 'ut 3', 'ut-1', 'ut-2', 'ut-3', 
                                                're-u', 're u', 'reu', 'ut1', 'ut2', 'ut3']):
            col_map['marks_cols'].append(col)
            
            subject = extract_subject_from_column(col)
            if subject not in col_map['subjects']:
                col_map['subjects'][subject] = []
            col_map['subjects'][subject].append(col)
    
    return col_map


def extract_subject_from_column(col_name):
    """Extract subject name from column"""
    col_str = str(col_name)
    subject = re.sub(r'(?i)\s*(ut[\s\-]*\d|re[\s\-]*u[\s\-]*\d?)\s*', '', col_str)
    subject = subject.strip()
    
    if not subject or len(subject) < 3:
        match = re.search(r'(?i)(subject\s*\d+)', col_str)
        if match:
            subject = match.group(1)
        else:
            subject = "Subject"
    
    return subject


def parse_test_type(col_name):
    """Identify test type"""
    col_upper = str(col_name).upper()
    
    if 'UT 1' in col_upper or 'UT-1' in col_upper or 'UT1' in col_upper:
        return 'UT-1'
    elif 'UT 2' in col_upper or 'UT-2' in col_upper or 'UT2' in col_upper:
        return 'UT-2'
    elif 'UT 3' in col_upper or 'UT-3' in col_upper or 'UT3' in col_upper:
        return 'UT-3'
    elif 'RE-U' in col_upper or 'RE U' in col_upper or 'REU' in col_upper:
        return 'RE-U'
    else:
        return 'Other'


def safe_numeric_conversion(series):
    """Safely convert series to numeric"""
    series = series.replace({
        'Absent': np.nan, 'absent': np.nan, 'ABSENT': np.nan,
        'AB': np.nan, 'ab': np.nan, 'Ab': np.nan,
        '-': np.nan, '—': np.nan, 'NA': np.nan, 'na': np.nan,
        'N/A': np.nan, 'n/a': np.nan, '': np.nan
    })
    return pd.to_numeric(series, errors='coerce')


def calculate_grade(percentage):
    """Calculate grade from percentage"""
    if pd.isna(percentage):
        return 'N/A'
    
    pct = float(percentage)
    if pct >= 90:
        return 'A+ Outstanding'
    elif pct >= 80:
        return 'A Excellent'
    elif pct >= 70:
        return 'B+ Very Good'
    elif pct >= 60:
        return 'B Good'
    elif pct >= 50:
        return 'C Average'
    elif pct >= 40:
        return 'D Pass'
    else:
        return 'F Fail'


def process_student_data(df, col_map, max_marks_per_test=25):
    """Process and enrich student data"""
    
    for col in col_map['marks_cols']:
        df[col] = safe_numeric_conversion(df[col])
    
    marks_data = df[col_map['marks_cols']].fillna(0)
    df['Total_Marks'] = marks_data.sum(axis=1)
    df['Average_Marks'] = marks_data.mean(axis=1)
    df['Max_Possible'] = len(col_map['marks_cols']) * max_marks_per_test
    df['Percentage'] = np.where(
        df['Max_Possible'] > 0,
        (df['Total_Marks'] / df['Max_Possible']) * 100,
        0
    )
    df['Grade'] = df['Percentage'].apply(calculate_grade)
    df['Rank'] = df['Total_Marks'].rank(ascending=False, method='min').astype(int)
    df['Pass_Status'] = df['Percentage'].apply(lambda x: '✓ Pass' if x >= 40 else '✗ Fail')
    
    for subject, cols in col_map['subjects'].items():
        subject_data = df[cols].fillna(0)
        df[f'{subject}_Total'] = subject_data.sum(axis=1)
        df[f'{subject}_Average'] = subject_data.mean(axis=1)
        df[f'{subject}_Max'] = len(cols) * max_marks_per_test
        df[f'{subject}_Percentage'] = np.where(
            df[f'{subject}_Max'] > 0,
            (df[f'{subject}_Total'] / df[f'{subject}_Max']) * 100,
            0
        )
    
    return df


def create_marks_distribution_chart(df):
    """Create detailed marks distribution"""
    st.markdown('<div class="section-header">📊 Marks Distribution Analysis</div>', unsafe_allow_html=True)
    
    # Define mark ranges
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    
    df['Marks_Range'] = pd.cut(df['Percentage'], bins=bins, labels=labels, include_lowest=True)
    range_counts = df['Marks_Range'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = go.Figure()
        colors = ['#dc3545' if i < 4 else '#ffc107' if i < 6 else '#28a745' 
                  for i in range(len(range_counts))]
        
        fig.add_trace(go.Bar(
            x=range_counts.index.astype(str),
            y=range_counts.values,
            text=range_counts.values,
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>%{x}%</b><br>Students: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Students by Marks Range',
            xaxis_title='Percentage Range',
            yaxis_title='Number of Students',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed stats cards
        fail_count = (df['Percentage'] < 40).sum()
        pass_count = ((df['Percentage'] >= 40) & (df['Percentage'] < 60)).sum()
        good_count = ((df['Percentage'] >= 60) & (df['Percentage'] < 80)).sum()
        excellent_count = (df['Percentage'] >= 80).sum()
        
        st.markdown(f"""
        <div class="danger-box">
            <div class="stat-number">{fail_count}</div>
            <div class="stat-label">Failed Students (Below 40%)</div>
            <div style="margin-top: 10px; color: #666;">{(fail_count/len(df)*100):.1f}% of class</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-box">
            <div class="stat-number">{pass_count}</div>
            <div class="stat-label">Average Students (40-60%)</div>
            <div style="margin-top: 10px; color: #666;">{(pass_count/len(df)*100):.1f}% of class</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="success-box">
            <div class="stat-number">{excellent_count}</div>
            <div class="stat-label">Excellent Students (80%+)</div>
            <div style="margin-top: 10px; color: #666;">{(excellent_count/len(df)*100):.1f}% of class</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed table
    st.markdown("#### 📋 Detailed Distribution")
    
    dist_data = []
    for range_label in labels:
        count = range_counts.get(range_label, 0)
        percentage = (count / len(df) * 100) if len(df) > 0 else 0
        
        if range_label in ['0-10', '10-20', '20-30', '30-40']:
            status = '🔴 Critical'
            color = '#ffebee'
        elif range_label in ['40-50', '50-60']:
            status = '🟡 Average'
            color = '#fff8e1'
        elif range_label in ['60-70', '70-80']:
            status = '🟢 Good'
            color = '#e8f5e9'
        else:
            status = '🌟 Excellent'
            color = '#e3f2fd'
        
        dist_data.append({
            'Marks Range': range_label + '%',
            'Students': count,
            'Percentage': f"{percentage:.1f}%",
            'Status': status
        })
    
    dist_df = pd.DataFrame(dist_data)
    st.dataframe(
        dist_df.style.set_properties(**{'background-color': '#f8f9fa', 'border': '1px solid #dee2e6'}),
        use_container_width=True,
        height=400
    )


def create_performance_overview(df, col_map, sheet_name="Overall"):
    """Create comprehensive performance overview"""
    
    st.markdown(f'<div class="section-header">📈 Performance Overview - {sheet_name}</div>', unsafe_allow_html=True)
    
    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Students</div>
            <div class="stat-number">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = df['Average_Marks'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Average Score</div>
            <div class="stat-number">{avg_score:.1f}/25</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_pct = df['Percentage'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Class Average</div>
            <div class="stat-number">{avg_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pass_rate = (df['Percentage'] >= 40).sum() / len(df) * 100
        color = '#28a745' if pass_rate >= 70 else '#ffc107' if pass_rate >= 50 else '#dc3545'
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Pass Rate</div>
            <div class="stat-number" style="color: {color};">{pass_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        highest = df['Total_Marks'].max()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Highest Score</div>
            <div class="stat-number">{highest:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Class Average by Subject (Marks-centric)
    if col_map['subjects']:
        st.markdown("#### 🎯 Class Average by Subject (Marks)")
        subject_avgs = []
        for subject in col_map['subjects'].keys():
            subject_avgs.append({
                'Subject': subject,
                'Average Marks': df[f'{subject}_Average'].mean()
            })
        subject_avg_df = pd.DataFrame(subject_avgs)

        fig = go.Figure(data=[go.Bar(
            x=subject_avg_df['Subject'],
            y=subject_avg_df['Average Marks'],
            text=subject_avg_df['Average Marks'].round(2),
            textposition='outside',
            marker_color='#4c78a8'
        )])
        fig.update_layout(
            height=420,
            xaxis_tickangle=-45,
            yaxis_title='Average Marks (out of 25)',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers
    st.markdown("#### 🏆 Top 15 Performers")
    top15 = df.nlargest(15, 'Total_Marks')
    display_cols = ['Rank']
    if col_map['name']:
        display_cols.append(col_map['name'])
    display_cols.extend(['Total_Marks', 'Percentage', 'Grade'])
    
    st.dataframe(
        top15[display_cols].style.format({'Percentage': '{:.2f}%', 'Total_Marks': '{:.0f}'})
        .background_gradient(subset=['Total_Marks'], cmap='Greens')
        .set_properties(**{'text-align': 'center'}),
        height=400,
        use_container_width=True
    )


def create_subject_analysis(df, col_map, sheet_name="Overall"):
    """Subject-wise analysis focused on UT-1 / UT-2 and toppers"""
    st.markdown(f'<div class="section-header">📚 Subject-wise Analysis - {sheet_name}</div>', unsafe_allow_html=True)

    if not col_map['subjects']:
        st.warning("No subjects detected")
        return

    # Build class averages by subject for UT-1 and UT-2
    subjects = list(col_map['subjects'].keys())
    ut1_avgs, ut2_avgs = [], []

    def subject_ut_avg(subject, ut_label):
        cols = [c for c in col_map['subjects'][subject] if parse_test_type(c) == ut_label]
        if not cols:
            return np.nan
        return df[cols].fillna(0).mean(axis=1).mean()

    for subject in subjects:
        ut1_avgs.append(subject_ut_avg(subject, 'UT-1'))
        ut2_avgs.append(subject_ut_avg(subject, 'UT-2'))

    comp_df = pd.DataFrame({
        'Subject': subjects,
        'UT-1 Avg': ut1_avgs,
        'UT-2 Avg': ut2_avgs,
    })

    st.markdown("#### 📈 Class Average by Subject and Test")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='UT-1', x=comp_df['Subject'], y=comp_df['UT-1 Avg'], marker_color='#4c78a8', text=np.round(comp_df['UT-1 Avg'], 2), textposition='outside'))
    fig.add_trace(go.Bar(name='UT-2', x=comp_df['Subject'], y=comp_df['UT-2 Avg'], marker_color='#f28e2b', text=np.round(comp_df['UT-2 Avg'], 2), textposition='outside'))
    fig.update_layout(barmode='group', height=420, xaxis_tickangle=-45, yaxis_title='Average Marks (out of 25)', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # Select a subject for topper lists
    sel_subject = st.selectbox("Select Subject for Toppers", subjects, key=f"sub_top_{sheet_name}")

    name_col = col_map['name'] if col_map['name'] else None

    def build_toppers(ut_label):
        cols = [c for c in col_map['subjects'][sel_subject] if parse_test_type(c) == ut_label]
        if not cols:
            return pd.DataFrame(columns=['Student', 'Marks'])
        # If multiple columns for same subject+UT, average them per student
        series = df[cols].fillna(0).mean(axis=1)
        out_df = pd.DataFrame({
            'Student': df[name_col].astype(str) if name_col else df.index.astype(str),
            'Marks': series
        })
        return out_df.sort_values('Marks', ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏅 UT-1 Toppers")
        t1 = build_toppers('UT-1')
        st.dataframe(
            t1.style.format({'Marks': '{:.2f}'}).background_gradient(subset=['Marks'], cmap='Blues'),
            use_container_width=True, height=340
        )
    with col2:
        st.markdown("#### 🏅 UT-2 Toppers")
        t2 = build_toppers('UT-2')
        st.dataframe(
            t2.style.format({'Marks': '{:.2f}'}).background_gradient(subset=['Marks'], cmap='Oranges'),
            use_container_width=True, height=340
        )

    # Overall subject toppers using total subject marks (across UTs)
    st.markdown("#### 🥇 Overall Subject Toppers (Total Marks across UTs)")
    overall_cols = col_map['subjects'][sel_subject]
    overall_series = df[overall_cols].fillna(0).sum(axis=1)
    overall_df = pd.DataFrame({
        'Student': df[name_col].astype(str) if name_col else df.index.astype(str),
        'Total Marks': overall_series
    }).sort_values('Total Marks', ascending=False).head(15)
    st.dataframe(
        overall_df.style.format({'Total Marks': '{:.0f}'}).background_gradient(subset=['Total Marks'], cmap='Greens'),
        use_container_width=True, height=380
    )


def create_test_comparison(df, col_map, sheet_name="Overall"):
    """Enhanced test comparison analysis"""
    st.markdown(f'<div class="section-header">📝 Unit Test Comparison - {sheet_name}</div>', unsafe_allow_html=True)
    
    test_groups = {}
    for col in col_map['marks_cols']:
        test_type = parse_test_type(col)
        if test_type not in test_groups:
            test_groups[test_type] = []
        test_groups[test_type].append(col)
    
    test_stats = []
    for test_type, cols in test_groups.items():
        test_data = df[cols].fillna(0)
        test_stats.append({
            'Test': test_type,
            'Average': test_data.mean().mean(),
            'Median': test_data.median().median(),
            'Std Dev': test_data.std().mean(),
            'Tests': len(cols)
        })
    
    test_df = pd.DataFrame(test_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Average Performance by Test Type")
        fig = go.Figure(data=[go.Bar(
            x=test_df['Test'],
            y=test_df['Average'],
            text=test_df['Average'].round(2),
            textposition='outside',
            marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
            hovertemplate='<b>%{x}</b><br>Average: %{y:.2f}/25<extra></extra>'
        )])
        
        fig.update_layout(
            height=400,
            yaxis_title='Average Marks',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Test Performance Distribution")
        fig = go.Figure()
        
        for test_type, cols in test_groups.items():
            test_data = df[cols].fillna(0).mean(axis=1)
            fig.add_trace(go.Box(
                y=test_data,
                name=test_type,
                boxmean='sd'
            ))
        
        fig.update_layout(
            height=400,
            yaxis_title='Marks',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # UT Progression Analysis
    ut_cols = {
        'UT-1': [c for c in col_map['marks_cols'] if parse_test_type(c) == 'UT-1'],
        'UT-2': [c for c in col_map['marks_cols'] if parse_test_type(c) == 'UT-2'],
        'UT-3': [c for c in col_map['marks_cols'] if parse_test_type(c) == 'UT-3']
    }
    
    if ut_cols['UT-1'] and ut_cols['UT-2']:
        st.markdown("#### 📈 Student Progression (UT-1 to UT-2)")
        
        df['UT1_Avg'] = df[ut_cols['UT-1']].fillna(0).mean(axis=1)
        df['UT2_Avg'] = df[ut_cols['UT-2']].fillna(0).mean(axis=1)
        df['Improvement'] = df['UT2_Avg'] - df['UT1_Avg']
        
        improved = (df['Improvement'] > 0).sum()
        declined = (df['Improvement'] < 0).sum()
        stable = (df['Improvement'] == 0).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="success-box">
                <div class="stat-number">{improved}</div>
                <div class="stat-label">Improved</div>
                <div style="margin-top: 5px; color: #666;">{(improved/len(df)*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="danger-box">
                <div class="stat-number">{declined}</div>
                <div class="stat-label">Declined</div>
                <div style="margin-top: 5px; color: #666;">{(declined/len(df)*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="warning-box">
                <div class="stat-number">{stable}</div>
                <div class="stat-label">Stable</div>
                <div style="margin-top: 5px; color: #666;">{(stable/len(df)*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_improvement = df['Improvement'].mean()
            color = 'success-box' if avg_improvement > 0 else 'danger-box'
            st.markdown(f"""
            <div class="{color}">
                <div class="stat-number">{avg_improvement:+.2f}</div>
                <div class="stat-label">Avg Change</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Improvement scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['UT1_Avg'],
            y=df['UT2_Avg'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['Improvement'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Improvement"),
                line=dict(width=1, color='white')
            ),
            text=df[col_map['name']] if col_map['name'] else df.index,
            hovertemplate='<b>%{text}</b><br>UT-1: %{x:.2f}<br>UT-2: %{y:.2f}<extra></extra>'
        ))
        
        # Add diagonal line
        max_val = max(df['UT1_Avg'].max(), df['UT2_Avg'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='UT-1 vs UT-2 Performance',
            xaxis_title='UT-1 Average',
            yaxis_title='UT-2 Average',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


def create_insights(df, col_map, sheet_name="Overall"):
    """Generate insights (marks-centric)"""
    st.markdown(f'<div class="section-header">💡 Key Insights - {sheet_name}</div>', unsafe_allow_html=True)

    mean_marks = df['Average_Marks'].mean()
    median_marks = df['Average_Marks'].median()
    std_marks = df['Average_Marks'].std()
    highest_total = df['Total_Marks'].max()
    q1 = df['Total_Marks'].quantile(0.25)
    q3 = df['Total_Marks'].quantile(0.75)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class=\"stat-card\">
            <div class=\"stat-label\">Mean Marks</div>
            <div class=\"stat-number\">{mean_marks:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class=\"stat-card\">
            <div class=\"stat-label\">Median Marks</div>
            <div class=\"stat-number\">{median_marks:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class=\"stat-card\">
            <div class=\"stat-label\">Std Dev</div>
            <div class=\"stat-number\">{std_marks:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class=\"stat-card\">
            <div class=\"stat-label\">Highest Total</div>
            <div class=\"stat-number\">{highest_total:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class=\"stat-card\">
            <div class=\"stat-label\">IQR (Total)</div>
            <div class=\"stat-number\">{(q3-q1):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Subject performance heatmap (marks)
    if col_map['subjects']:
        st.markdown("#### 🔥 Subject Performance Heatmap (Average Marks)")

        subject_matrix = []
        for subject in col_map['subjects'].keys():
            subject_matrix.append(df[f'{subject}_Average'].values)

        fig = go.Figure(data=go.Heatmap(
            z=subject_matrix,
            x=[f"Student {i+1}" for i in range(len(df))],
            y=list(col_map['subjects'].keys()),
            colorscale='Viridis',
            zmin=0,
            zmax=25,
            hovertemplate='Subject: %{y}<br>Student: %{x}<br>Avg: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            height=420,
            xaxis_title='Students',
            yaxis_title='Subjects',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


def create_individual_reports(df, col_map, sheet_name="Overall"):
    """Enhanced individual student reports"""
    st.markdown(f'<div class="section-header">👤 Individual Student Reports - {sheet_name}</div>', unsafe_allow_html=True)
    
    if not col_map['name']:
        st.warning("Student name column not found")
        return
    
    students = sorted(df[col_map['name']].astype(str).unique().tolist())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("🔍 Select Student", students, key=f"student_select_{sheet_name}")
    with col2:
        search_by_rank = st.number_input("Or Search by Rank", min_value=1, max_value=len(df), value=1, key=f"rank_search_{sheet_name}")
        if st.button("Go to Rank", key=f"rank_btn_{sheet_name}"):
            rank_student = df[df['Rank'] == search_by_rank]
            if not rank_student.empty and col_map['name']:
                selected = str(rank_student.iloc[0][col_map['name']])
    
    student_data = df[df[col_map['name']].astype(str) == selected].iloc[0]
    
    # Student Header Card
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color:white;">📋 {selected}</h2>
        <p style="margin:5px 0 0 0; opacity:0.9;">Performance Report</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        rank_color = '#28a745' if student_data['Rank'] <= 10 else '#ffc107' if student_data['Rank'] <= 30 else '#dc3545'
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Class Rank</div>
            <div class="stat-number" style="color: {rank_color};">#{student_data['Rank']}</div>
            <div style="font-size:12px; color:#666;">out of {len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Marks</div>
            <div class="stat-number">{student_data['Total_Marks']:.0f}</div>
            <div style="font-size:12px; color:#666;">out of {student_data['Max_Possible']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Average Marks</div>
            <div class="stat-number">{student_data['Average_Marks']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Grade</div>
            <div class="stat-number" style="font-size:20px;">{student_data['Grade']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        status_color = 'success-box' if '✓' in student_data['Pass_Status'] else 'danger-box'
        st.markdown(f"""
        <div class="{status_color}">
            <div class="stat-label">Status</div>
            <div class="stat-number" style="font-size:24px;">{student_data['Pass_Status']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Subject Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Subject-wise Performance (Marks)")
        subject_perf = []
        for subject in col_map['subjects'].keys():
            subject_perf.append({
                'Subject': subject,
                'Average': student_data[f'{subject}_Average']
            })

        subject_df = pd.DataFrame(subject_perf)

        fig = go.Figure(data=[go.Bar(
            x=subject_df['Subject'],
            y=subject_df['Average'],
            text=subject_df['Average'].round(2),
            textposition='outside',
            marker_color='#4c78a8'
        )])

        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            yaxis_title='Average Marks',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Performance Comparison")
        
        # Create comparison with class average
        comparison_data = []
        for subject in col_map['subjects'].keys():
            class_avg = df[f'{subject}_Average'].mean()
            student_avg = student_data[f'{subject}_Average']
            comparison_data.append({
                'Subject': subject,
                'Student': student_avg,
                'Class Average': class_avg
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Student',
            x=comp_df['Subject'],
            y=comp_df['Student'],
            marker_color='#667eea',
            text=comp_df['Student'].round(2),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Class Average',
            x=comp_df['Subject'],
            y=comp_df['Class Average'],
            marker_color='#764ba2',
            text=comp_df['Class Average'].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            height=400,
            xaxis_tickangle=-45,
            yaxis_title='Average Marks',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed marks table
    st.markdown("#### 📊 Detailed Marks Breakdown")
    marks_data = []
    for col in col_map['marks_cols']:
        test_type = parse_test_type(col)
        subject = extract_subject_from_column(col)
        marks = student_data[col] if not pd.isna(student_data[col]) else 0
        
        marks_data.append({
            'Subject': subject,
            'Test': test_type,
            'Marks': marks,
            'Out of': 25
        })
    
    marks_df = pd.DataFrame(marks_data)
    st.dataframe(
        marks_df.style.format({
            'Marks': '{:.0f}'
        }).background_gradient(subset=['Marks'], cmap='RdYlGn', vmin=0, vmax=25)
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        height=350
    )


def create_subjectwise_selection(df, col_map, sheet_name="Overall"):
    """Tab for division and subject-based selection with mark thresholds"""
    st.markdown(f'<div class="section-header">📘 Subjectwise Selection - {sheet_name}</div>', unsafe_allow_html=True)

    if not col_map['subjects']:
        st.warning("No subjects detected")
        return

    subjects = list(col_map['subjects'].keys())
    name_col = col_map['name']
    prn_col = col_map['prn']
    div_col = col_map['division']

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        sel_subject = st.selectbox("Select Subject", subjects, key=f"sub_sel_{sheet_name}")
    with col2:
        # Determine available tests for this subject
        available_tests = sorted(set(parse_test_type(c) for c in col_map['subjects'][sel_subject]))
        # Prioritize UT-1 then UT-2
        default_test = 'UT-1' if 'UT-1' in available_tests else available_tests[0]
        sel_test = st.selectbox("Select Test", available_tests, index=available_tests.index(default_test), key=f"test_sel_{sheet_name}")
    with col3:
        if div_col:
            divisions = ['All'] + sorted(df[div_col].dropna().astype(str).unique().tolist())
            sel_div = st.selectbox("Division", divisions, key=f"div_sel_{sheet_name}")
        else:
            sel_div = 'All'

    # Build selected marks (if multiple columns for same subject/test, average them)
    sel_cols = [c for c in col_map['subjects'][sel_subject] if parse_test_type(c) == sel_test]
    if not sel_cols:
        st.info("No marks found for the selected subject/test.")
        return

    work_df = df.copy()
    if div_col and sel_div != 'All':
        work_df = work_df[work_df[div_col].astype(str) == sel_div]

    marks_series = work_df[sel_cols].fillna(0).mean(axis=1)
    display_df = pd.DataFrame({
        'PRN': work_df[prn_col].astype(str) if prn_col else work_df.index.astype(str),
        'Name': work_df[name_col].astype(str) if name_col else work_df.index.astype(str),
        'Division': work_df[div_col].astype(str) if div_col else '',
        'Marks (out of 25)': marks_series.round(2)
    })

    st.markdown("#### 📄 Filtered Students (Original Data)")
    st.dataframe(
        display_df,
        use_container_width=True,
        height=350
    )

    # Categories
    le_10 = display_df[display_df['Marks (out of 25)'] <= 10]
    ge_15 = display_df[display_df['Marks (out of 25)'] >= 15]
    gt_20 = display_df[display_df['Marks (out of 25)'] > 20]

    # Counts chart
    st.markdown("#### 📊 Category Counts")
    cat_labels = ['<= 10', '15 to 25', '> 20']
    cat_vals = [len(le_10), len(ge_15), len(gt_20)]
    fig = go.Figure(data=[go.Bar(x=cat_labels, y=cat_vals, text=cat_vals, textposition='outside', marker_color=['#e15759','#4c78a8','#59a14f'])])
    fig.update_layout(height=320, template='plotly_white', xaxis_title='', yaxis_title='Students', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Lists
    st.markdown("#### 🔻 Students with 10 or below (<= 40%)")
    st.dataframe(le_10, use_container_width=True, height=220)

    st.markdown("#### 🔷 Students with 15 to 25 marks")
    st.dataframe(ge_15, use_container_width=True, height=220)

    st.markdown("#### 🟢 Students with > 20 marks")
    st.dataframe(gt_20, use_container_width=True, height=220)


def create_sheet_comparison(all_sheets_data):
    """Compare performance across multiple sheets"""
    st.markdown('<div class="section-header">📑 Multi-Sheet Comparison</div>', unsafe_allow_html=True)
    
    comparison_stats = []
    for sheet_name, (df, col_map) in all_sheets_data.items():
        comparison_stats.append({
            'Sheet': sheet_name,
            'Students': len(df),
            'Avg Marks': df['Average_Marks'].mean(),
            'Highest Total': df['Total_Marks'].max(),
            'Median Marks': df['Average_Marks'].median()
        })
    
    comp_df = pd.DataFrame(comparison_stats)
    
    # Sheet comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Average Marks by Sheet")
        fig = go.Figure(data=[go.Bar(
            x=comp_df['Sheet'],
            y=comp_df['Avg Marks'],
            text=comp_df['Avg Marks'].round(1),
            textposition='outside',
            marker_color='#4c78a8'
        )])
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            yaxis_title='Average Marks',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Median Marks by Sheet")
        fig = go.Figure(data=[go.Bar(
            x=comp_df['Sheet'],
            y=comp_df['Median Marks'],
            text=comp_df['Median Marks'].round(1),
            textposition='outside',
            marker_color='#72b7b2'
        )])
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            yaxis_title='Median Marks',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("#### 📊 Detailed Sheet Statistics")
    st.dataframe(
        comp_df.style.format({
            'Avg Marks': '{:.2f}',
            'Median Marks': '{:.2f}',
            'Highest Total': '{:.0f}'
        }).background_gradient(subset=['Avg Marks', 'Median Marks'], cmap='Blues')
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True
    )


def main():
    # Header
    st.markdown('<div class="main-header">🎓 Student Performance Analytics Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced multi-sheet analysis with stunning visualizations</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("📁 Upload Excel/CSV", type=['xlsx', 'xls', 'csv'])
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        max_marks = st.number_input("Max marks per test", 1, 100, 25)
        
        st.markdown("---")
        st.markdown("### 🎨 Display Options")
        show_all_sheets = st.checkbox("📊 Analyze all sheets separately", value=True)
        show_comparison = st.checkbox("📑 Show sheet comparison", value=True)
        show_debug = st.checkbox("🔍 Show debug info", value=False)
        
        st.markdown("---")
        st.markdown("""
        <div class="insight-box">
        <strong>✨ Features:</strong><br>
        • Multi-sheet analysis<br>
        • Subject-wise toppers (UT-1 / UT-2)<br>
        • Class averages by subject and test<br>
        • Individual student reports<br>
        • Sheet-to-sheet comparison (marks-centric)<br>
        • Export functionality
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Made with ❤️ using Streamlit**")
    
    if uploaded_file is None:
        # Welcome screen
        st.markdown("""
        <div class="insight-box">
            <h3>👋 Welcome to Student Analytics Pro!</h3>
            <p>Upload your Excel file to get started with comprehensive performance analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>📊 Visualizations</h4>
                <p>Beautiful charts and graphs for every metric</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>📚 Multi-Sheet</h4>
                <p>Analyze multiple sheets with comparison</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>👤 Individual</h4>
                <p>Detailed student-wise performance reports</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Supported Format")
        st.markdown("""
        Your Excel file should contain:
        - **Student Name/PRN columns**
        - **Subject columns** with test marks (UT-1, UT-2, UT-3, RE-U)
        - **Multiple sheets** (optional) - will be analyzed separately
        - **Multi-row headers** are fully supported!
        """)
        
        return
    
    try:
        file_bytes = uploaded_file.read()
        
        with st.spinner("📂 Processing file..."):
            # Read first sheet to get all sheet names
            df_first, sheet_names = read_excel_with_multirow_headers(file_bytes, uploaded_file.name)
            
            if df_first is None:
                st.error("❌ Failed to read file")
                return
            
            # Process all sheets if requested
            all_sheets_data = {}
            
            if show_all_sheets and sheet_names and len(sheet_names) > 1:
                st.success(f"✅ Found {len(sheet_names)} sheets")
                
                # Display sheet badges
                badge_html = "".join([f'<span class="sheet-badge">{sheet}</span>' for sheet in sheet_names])
                st.markdown(f'<div style="text-align:center; margin:20px 0;">{badge_html}</div>', unsafe_allow_html=True)
                
                # Process each sheet
                for sheet_name in sheet_names:
                    df, _ = read_excel_with_multirow_headers(file_bytes, uploaded_file.name, sheet_name)
                    
                    if df is not None and len(df) > 0:
                        col_map = detect_columns(df)
                        
                        if col_map['marks_cols']:
                            df = process_student_data(df, col_map, max_marks)
                            all_sheets_data[sheet_name] = (df, col_map)
                
                if not all_sheets_data:
                    st.error("❌ No valid data found in any sheet")
                    return
                
                st.success(f"✅ Successfully processed {len(all_sheets_data)} sheets with student data")
            else:
                # Process single sheet
                col_map = detect_columns(df_first)
                
                if not col_map['marks_cols']:
                    st.error("❌ No marks columns detected!")
                    st.info("**Detected columns:** " + ", ".join(df_first.columns.tolist()))
                    st.info("💡 Looking for columns with: UT-1, UT-2, UT-3, RE-U")
                    return
                
                df_first = process_student_data(df_first, col_map, max_marks)
                all_sheets_data['Main'] = (df_first, col_map)
                st.success(f"✅ Processed {len(df_first)} students | {len(col_map['subjects'])} subjects | {len(col_map['marks_cols'])} tests")
            
            if show_debug:
                with st.expander("🔍 Debug Information"):
                    for sheet_name, (df, col_map) in all_sheets_data.items():
                        st.write(f"**Sheet: {sheet_name}**")
                        st.write("**Column Map:**", col_map)
                        st.write("**Data Preview:**")
                        st.dataframe(df.head())
                        st.markdown("---")
        
        # Main Analysis Section
        st.markdown("---")
        
        if len(all_sheets_data) > 1 and show_comparison:
            # Sheet Comparison View
            create_sheet_comparison(all_sheets_data)
            st.markdown("---")
        
        # Create tabs for each sheet
        if len(all_sheets_data) == 1:
            # Single sheet - show all tabs
            sheet_name, (df, col_map) = list(all_sheets_data.items())[0]
            
            tabs = st.tabs([
                "📈 Overview", 
                "📚 Subjects", 
                "📝 UT Analysis", 
                "📘 Subjectwise Selection",
                "💡 Insights", 
                "👤 Individual", 
                "💾 Download"
            ])
            
            with tabs[0]:
                create_performance_overview(df, col_map, sheet_name)

            with tabs[1]:
                create_subject_analysis(df, col_map, sheet_name)
            
            with tabs[2]:
                create_test_comparison(df, col_map, sheet_name)
            
            with tabs[3]:
                create_subjectwise_selection(df, col_map, sheet_name)

            with tabs[4]:
                create_insights(df, col_map, sheet_name)
            
            with tabs[5]:
                create_individual_reports(df, col_map, sheet_name)
            
            with tabs[6]:
                st.markdown('<div class="section-header">💾 Download Reports</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📄 Complete Analysis")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Full Data",
                        csv,
                        f"analysis_{sheet_name}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("#### 🏆 Rankings")
                    rank_cols = ['Rank']
                    if col_map['name']:
                        rank_cols.append(col_map['name'])
                    rank_cols.extend(['Total_Marks', 'Percentage', 'Grade'])
                    
                    csv_rank = df.sort_values('Rank')[rank_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Rankings",
                        csv_rank,
                        f"rankings_{sheet_name}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    st.markdown("#### ⚠️ Failed Students")
                    failed = df[df['Percentage'] < 40]
                    if len(failed) > 0:
                        fail_cols = ['Rank']
                        if col_map['name']:
                            fail_cols.append(col_map['name'])
                        fail_cols.extend(['Total_Marks', 'Percentage'])
                        
                        csv_fail = failed[fail_cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Failed List",
                            csv_fail,
                            f"failed_{sheet_name}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.success("🎉 No failed students!")
        
        else:
            # Multiple sheets - create separate sections
            for sheet_name, (df, col_map) in all_sheets_data.items():
                st.markdown(f"## 📑 {sheet_name}")
                
                tabs = st.tabs([
                    "📈 Overview", 
                    "📚 Subjects", 
                    "📝 UT Analysis", 
                    "📘 Subjectwise Selection",
                    "💡 Insights", 
                    "👤 Individual"
                ])
                
                with tabs[0]:
                    create_performance_overview(df, col_map, sheet_name)

                with tabs[1]:
                    create_subject_analysis(df, col_map, sheet_name)
                
                with tabs[2]:
                    create_test_comparison(df, col_map, sheet_name)
                
                with tabs[3]:
                    create_subjectwise_selection(df, col_map, sheet_name)

                with tabs[4]:
                    create_insights(df, col_map, sheet_name)
                
                with tabs[5]:
                    create_individual_reports(df, col_map, sheet_name)
                
                st.markdown("---")
            
            # Combined downloads section
            st.markdown('<div class="section-header">💾 Download All Reports</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📄 Combined Data")
                # Combine all sheets
                combined_dfs = []
                for sheet_name, (df, col_map) in all_sheets_data.items():
                    df_copy = df.copy()
                    df_copy['Sheet'] = sheet_name
                    combined_dfs.append(df_copy)
                
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                csv_combined = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download All Sheets Combined",
                    csv_combined,
                    "all_sheets_combined.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🏆 All Rankings")
                rank_dfs = []
                for sheet_name, (df, col_map) in all_sheets_data.items():
                    rank_cols = ['Rank']
                    if col_map['name']:
                        rank_cols.append(col_map['name'])
                    rank_cols.extend(['Total_Marks', 'Percentage', 'Grade'])
                    
                    df_rank = df.sort_values('Rank')[rank_cols].copy()
                    df_rank['Sheet'] = sheet_name
                    rank_dfs.append(df_rank)
                
                combined_ranks = pd.concat(rank_dfs, ignore_index=True)
                csv_ranks = combined_ranks.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download All Rankings",
                    csv_ranks,
                    "all_rankings.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("#### ⚠️ All Failed Students")
                fail_dfs = []
                for sheet_name, (df, col_map) in all_sheets_data.items():
                    failed = df[df['Percentage'] < 40]
                    if len(failed) > 0:
                        fail_cols = ['Rank']
                        if col_map['name']:
                            fail_cols.append(col_map['name'])
                        fail_cols.extend(['Total_Marks', 'Percentage'])
                        
                        df_fail = failed[fail_cols].copy()
                        df_fail['Sheet'] = sheet_name
                        fail_dfs.append(df_fail)
                
                if fail_dfs:
                    combined_fails = pd.concat(fail_dfs, ignore_index=True)
                    csv_fails = combined_fails.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Failed Students",
                        csv_fails,
                        "all_failed.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.success("🎉 No failed students in any sheet!")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>Student Performance Analytics Pro</strong></p>
            <p>Built with Streamlit | Powered by Plotly</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        if show_debug:
            st.code(traceback.format_exc())
            st.error("Please check your file format and try again.")


if __name__ == "__main__":
    main()