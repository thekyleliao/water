import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Beach Water Quality Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌊 Beach Water Quality Analysis Dashboard")
st.markdown("""
    This dashboard provides insights into beach water quality monitoring data, focusing on Enterococci levels.
    Use the sidebar to navigate between different views and filter the data.
""")

# NYC Beach Coordinates
BEACH_COORDINATES = {
    'Coney Island': {'lat': 40.575, 'lon': -73.983},
    'Kingsborough Community College': {'lat': 40.575863, 'lon': -73.936546},
    'Manhattan Beach': {'lat': 40.575088, 'lon': -73.946783},
    'Breezy Point': {'lat': 40.55, 'lon': -73.93},
    'Rockaway Beach': {'lat': 40.5855, 'lon': -73.8055},
    'Orchard Beach': {'lat': 40.865, 'lon': -73.784},
    'Midland Beach': {'lat': 40.57316, 'lon': -74.09459},
    'South Beach': {'lat': 40.590972, 'lon': -74.067639},
    'Cedar Grove Beach': {'lat': 40.536, 'lon': -74.148}
}

# Function to load and process data
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('Beach_Water_Samples_20250604.csv')
        
        # Convert Sample Date to datetime
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], format='%m/%d/%Y')
        
        # Handle missing values in Enterococci Results
        # Replace NaN values with 0 for samples below detection limit
        df.loc[df['Units or Notes'] == 'Result below detection limit', 'Enterococci Results'] = 0
        
        # Sort by date
        df = df.sort_values('Sample Date')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Select View", [
        "Beach Water Quality Overview", 
        "Detailed Beach Analysis", 
        "Beach Locations Map", 
        "AI Assistant",
        "Interactive Map & Chat",
        "Interactive Map with Comparison",
        "Beach Comparison Test 1"
    ])
    
    # Common filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range Filter")
    min_date = df['Sample Date'].min()
    max_date = df['Sample Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    mask = (df['Sample Date'].dt.date >= date_range[0]) & (df['Sample Date'].dt.date <= date_range[1])
    filtered_df = df[mask]
    
    # Safe limit for Enterococci
    safe_limit = st.sidebar.number_input("Enterococci Safe Limit (MPN/100ml)", value=104, min_value=0)
    
    if view == "Beach Water Quality Overview":
        st.header("Beach Water Quality Overview")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Enterococci Level", 
                     f"{filtered_df['Enterococci Results'].mean():.2f} MPN/100ml")
        with col2:
            st.metric("Maximum Enterococci Level", 
                     f"{filtered_df['Enterococci Results'].max():.2f} MPN/100ml")
        with col3:
            st.metric("Number of Exceedances", 
                     f"{len(filtered_df[filtered_df['Enterococci Results'] > safe_limit])}")
        
        # Beaches exceeding safe limits
        st.subheader("Beaches Exceeding Safe Limits")
        exceedances = filtered_df[filtered_df['Enterococci Results'] > safe_limit]
        if not exceedances.empty:
            fig = px.bar(
                exceedances.groupby('Beach Name')['Enterococci Results'].count().reset_index(),
                x='Beach Name',
                y='Enterococci Results',
                title="Number of Exceedances by Beach",
                labels={'Enterococci Results': 'Number of Exceedances'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No exceedances found in the selected date range.")
        
        # Temporal trends
        st.subheader("Temporal Trends")
        daily_avg = filtered_df.groupby('Sample Date')['Enterococci Results'].mean().reset_index()
        fig = px.line(
            daily_avg,
            x='Sample Date',
            y='Enterococci Results',
            title="Average Enterococci Levels Over Time"
        )
        fig.add_hline(y=safe_limit, line_dash="dash", line_color="red", annotation_text="Safe Limit")
        st.plotly_chart(fig, use_container_width=True)
        
    elif view == "Detailed Beach Analysis":
        st.header("Detailed Beach Analysis")
        
        # Beach selector
        selected_beaches = st.multiselect(
            "Select Beaches to Compare",
            options=sorted(df['Beach Name'].unique()),
            default=[sorted(df['Beach Name'].unique())[0]]
        )
        
        if selected_beaches:
            # Filter data for selected beaches
            beach_data = filtered_df[filtered_df['Beach Name'].isin(selected_beaches)]
            
            # Calculate daily averages for each beach
            daily_avg = beach_data.groupby(['Sample Date', 'Beach Name'])['Enterococci Results'].mean().reset_index()
            
            # Beach comparison over time
            st.subheader("Beach Comparison Over Time")
            fig = px.line(
                daily_avg,
                x='Sample Date',
                y='Enterococci Results',
                color='Beach Name',
                title="Enterococci Levels Over Time by Beach",
                labels={'Enterococci Results': 'Enterococci Level (MPN/100ml)'}
            )
            fig.add_hline(y=safe_limit, line_dash="dash", line_color="red", annotation_text="Safe Limit")
            st.plotly_chart(fig, use_container_width=True)
            
            # Beach comparison statistics
            st.subheader("Beach Comparison Statistics")
            beach_stats = beach_data.groupby('Beach Name').agg({
                'Enterococci Results': ['mean', 'max', 'min', 'std', 'count']
            }).round(2)
            beach_stats.columns = ['Average', 'Maximum', 'Minimum', 'Standard Deviation', 'Number of Samples']
            st.dataframe(beach_stats)
            
            # Box plot for beach comparison
            st.subheader("Distribution of Enterococci Levels by Beach")
            fig = px.box(
                beach_data,
                x='Beach Name',
                y='Enterococci Results',
                title="Distribution of Enterococci Levels by Beach",
                labels={'Enterococci Results': 'Enterococci Level (MPN/100ml)'}
            )
            fig.add_hline(y=safe_limit, line_dash="dash", line_color="red", annotation_text="Safe Limit")
            st.plotly_chart(fig, use_container_width=True)
            
            # Exceedance comparison
            st.subheader("Exceedance Comparison")
            exceedances = beach_data[beach_data['Enterococci Results'] > safe_limit]
            exceedance_counts = exceedances.groupby('Beach Name').size().reset_index(name='Exceedances')
            fig = px.bar(
                exceedance_counts,
                x='Beach Name',
                y='Exceedances',
                title="Number of Exceedances by Beach",
                color='Beach Name'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data table
            st.subheader("Raw Data")
            st.dataframe(beach_data.sort_values(['Beach Name', 'Sample Date'], ascending=[True, False]))
        else:
            st.info("Please select at least one beach to analyze.")
    elif view == "Beach Locations Map":
        st.header("Beach Locations Map")
        
        # Debug: Print unique beach names from dataset
        st.write("Beaches in dataset:", df['Beach Name'].unique())
        st.write("Beaches with coordinates:", list(BEACH_COORDINATES.keys()))
        
        # Create a map centered on NYC
        m = folium.Map(
            location=[40.7128, -74.0060],  # NYC coordinates
            zoom_start=10
        )
        
        # Get unique beach names from the dataset
        unique_beaches = df['Beach Name'].unique()
        
        # Add markers for each beach
        for beach_name in unique_beaches:
            # Try to find a matching beach name in our coordinates
            matching_beach = None
            for coord_name in BEACH_COORDINATES.keys():
                if coord_name.lower() in beach_name.lower() or beach_name.lower() in coord_name.lower():
                    matching_beach = coord_name
                    break
            
            if matching_beach:
                # Get the latest water quality data for this beach
                latest_data = df[df['Beach Name'] == beach_name].sort_values('Sample Date').iloc[-1]
                enterococci_level = latest_data['Enterococci Results']
                
                # Debug: Print marker information
                st.write(f"Adding marker for {beach_name} at {BEACH_COORDINATES[matching_beach]}")
                
                # Determine marker color based on water quality
                color = 'red' if enterococci_level > safe_limit else 'green'
                
                # Create popup content
                popup_content = f"""
                <b>{beach_name}</b><br>
                Latest Enterococci Level: {enterococci_level:.2f} MPN/100ml<br>
                Sample Date: {latest_data['Sample Date'].strftime('%Y-%m-%d')}
                """
                
                # Add marker directly to the map
                folium.Marker(
                    location=[BEACH_COORDINATES[matching_beach]['lat'], BEACH_COORDINATES[matching_beach]['lon']],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color=color),
                    tooltip=beach_name
                ).add_to(m)
            else:
                # Debug: Print beaches without coordinates
                st.write(f"No coordinates found for beach: {beach_name}")
        
        # Display the map
        st_folium(m, width=1000, height=600)
        
        # Add legend
        st.markdown("""
        ### Map Legend
        - 🟢 Green markers: Water quality within safe limits
        - 🔴 Red markers: Water quality exceeds safe limits
        """)
        
        # Add note about coordinates
        st.info("Note: Beach locations are approximate and based on general NYC beach locations.")
    elif view == "AI Assistant":
        st.header("AI Assistant")
        st.markdown("""
        Ask questions about the beach water quality data. The AI assistant has access to the dataset and can help you analyze trends, 
        find specific information, or explain the data.
        """)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about beach water quality..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare context from the dataset
            context = f"""
            You are a friendly beach water quality helper who loves teaching kids about water safety! 
            Keep your answers short and sweet - like a quick chat with a friend.

            Quick facts about our beaches:
            • {len(df)} water checks done
            • Checking since {df['Sample Date'].min().strftime('%Y-%m-%d')}
            • Beaches: {', '.join(df['Beach Name'].unique())}
            • Safe level: {safe_limit} MPN/100ml

            Latest checks:
            {df.sort_values('Sample Date').groupby('Beach Name').last()[['Sample Date', 'Enterococci Results']].to_string()}

            Fun facts:
            • {len(df[df['Enterococci Results'] > safe_limit])} times water wasn't clean enough
            • Average: {df['Enterococci Results'].mean():.2f} MPN/100ml
            • Highest: {df['Enterococci Results'].max():.2f} MPN/100ml

            Tips for talking to kids:
            • Keep it short and fun
            • Use simple words
            • Make it like telling a quick story
            • Compare numbers to things they know
            • Use our real data
            """
            
            # Prepare the full prompt with context
            full_prompt = f"""Context: {context}

            User question: {prompt}

            Remember: Keep your answer short and sweet! Use our data to give accurate answers, 
            but explain everything in a fun, simple way that a kid can understand quickly."""

            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Prepare API request
                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                headers = {
                    "Authorization": "Bearer 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",
                    "Accept": "text/event-stream"
                }

                payload = {
                    "model": "meta/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 1.00,
                    "frequency_penalty": 0.00,
                    "presence_penalty": 0.00,
                    "stream": True
                }

                try:
                    response = requests.post(invoke_url, headers=headers, json=payload)
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        full_response += content
                                        message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error communicating with AI service: {str(e)}")
                    st.info("Please try again later or rephrase your question.")
    elif view == "Interactive Map & Chat":
        st.header("Interactive Beach Map & AI Assistant")
        
        # Create two columns for the layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a map centered on NYC
            m = folium.Map(
                location=[40.7128, -74.0060],  # NYC coordinates
                zoom_start=10
            )
            
            # Get unique beach names from the dataset
            unique_beaches = df['Beach Name'].unique()
            
            # Add markers for each beach
            for beach_name in unique_beaches:
                # Try to find a matching beach name in our coordinates
                matching_beach = None
                for coord_name in BEACH_COORDINATES.keys():
                    if coord_name.lower() in beach_name.lower() or beach_name.lower() in coord_name.lower():
                        matching_beach = coord_name
                        break
                
                if matching_beach:
                    # Get the latest water quality data for this beach
                    latest_data = df[df['Beach Name'] == beach_name].sort_values('Sample Date').iloc[-1]
                    enterococci_level = latest_data['Enterococci Results']
                    
                    # Determine marker color based on water quality
                    color = 'red' if enterococci_level > safe_limit else 'green'
                    
                    # Create popup content
                    popup_content = f"""
                    <b>{beach_name}</b><br>
                    Latest Enterococci Level: {enterococci_level:.2f} MPN/100ml<br>
                    Sample Date: {latest_data['Sample Date'].strftime('%Y-%m-%d')}
                    """
                    
                    # Add marker directly to the map
                    folium.Marker(
                        location=[BEACH_COORDINATES[matching_beach]['lat'], BEACH_COORDINATES[matching_beach]['lon']],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(color=color),
                        tooltip=beach_name
                    ).add_to(m)
            
            # Display the map
            st_folium(m, width=800, height=400)
            
            # Add legend
            st.markdown("""
            ### Map Legend
            - 🟢 Green markers: Water quality within safe limits
            - 🔴 Red markers: Water quality exceeds safe limits
            """)
        
        with col2:
            st.markdown("### Beach Water Quality Helper")
            st.markdown("""
            Hi! I'm your friendly beach water quality helper! I can explain things in simple terms and help you understand 
            what makes our beaches safe and healthy. Click on a beach marker on the map and ask me anything!
            """)
            
            # Initialize chat history
            if "map_chat_messages" not in st.session_state:
                st.session_state.map_chat_messages = []
            
            # Display chat messages from history
            for message in st.session_state.map_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            if prompt := st.chat_input("Ask me about beach water quality..."):
                # Add user message to chat history
                st.session_state.map_chat_messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Prepare context from the dataset
                context = f"""
                You are a friendly beach water quality helper who loves teaching kids about water safety! 
                Keep your answers short and sweet - like a quick chat with a friend.

                Quick facts about our beaches:
                • {len(df)} water checks done
                • Checking since {df['Sample Date'].min().strftime('%Y-%m-%d')}
                • Beaches: {', '.join(df['Beach Name'].unique())}
                • Safe level: {safe_limit} MPN/100ml

                Latest checks:
                {df.sort_values('Sample Date').groupby('Beach Name').last()[['Sample Date', 'Enterococci Results']].to_string()}

                Fun facts:
                • {len(df[df['Enterococci Results'] > safe_limit])} times water wasn't clean enough
                • Average: {df['Enterococci Results'].mean():.2f} MPN/100ml
                • Highest: {df['Enterococci Results'].max():.2f} MPN/100ml

                Tips for talking to kids:
                • Keep it short and fun
                • Use simple words
                • Make it like telling a quick story
                • Compare numbers to things they know
                • Use our real data
                """
                
                # Prepare the full prompt with context
                full_prompt = f"""Context: {context}

                User question: {prompt}

                Remember: Keep your answer short and sweet! Use our data to give accurate answers, 
                but explain everything in a fun, simple way that a kid can understand quickly."""
                
                # Display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Prepare API request
                    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                    headers = {
                        "Authorization": "Bearer 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",
                        "Accept": "text/event-stream"
                    }
                    
                    payload = {
                        "model": "meta/llama-4-scout-17b-16e-instruct",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 1.00,
                        "frequency_penalty": 0.00,
                        "presence_penalty": 0.00,
                        "stream": True
                    }
                    
                    try:
                        response = requests.post(invoke_url, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode("utf-8").replace("data: ", ""))
                                    if "choices" in data and len(data["choices"]) > 0:
                                        content = data["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            full_response += content
                                            message_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    continue
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.map_chat_messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error communicating with AI service: {str(e)}")
                        st.info("Please try again later or rephrase your question.")
    elif view == "Interactive Map with Comparison":
        st.header("Interactive Beach Map with Comparison")
        
        # Create two columns for the layout
        map_col, chat_col = st.columns([2, 1])
        
        with map_col:
            # Create a map centered on NYC
            m = folium.Map(
                location=[40.7128, -74.0060],  # NYC coordinates
                zoom_start=10
            )
            
            # Get unique beach names from the dataset
            unique_beaches = df['Beach Name'].unique()
            
            # Add markers for each beach
            for beach_name in unique_beaches:
                # Try to find a matching beach name in our coordinates
                matching_beach = None
                for coord_name in BEACH_COORDINATES.keys():
                    if coord_name.lower() in beach_name.lower() or beach_name.lower() in coord_name.lower():
                        matching_beach = coord_name
                        break
                
                if matching_beach:
                    # Get the latest water quality data for this beach
                    latest_data = df[df['Beach Name'] == beach_name].sort_values('Sample Date').iloc[-1]
                    enterococci_level = latest_data['Enterococci Results']
                    
                    # Determine marker color based on water quality
                    color = 'red' if enterococci_level > safe_limit else 'green'
                    
                    # Create popup content
                    popup_content = f"""
                    <b>{beach_name}</b><br>
                    Latest Enterococci Level: {enterococci_level:.2f} MPN/100ml<br>
                    Sample Date: {latest_data['Sample Date'].strftime('%Y-%m-%d')}
                    """
                    
                    # Add marker directly to the map
                    folium.Marker(
                        location=[BEACH_COORDINATES[matching_beach]['lat'], BEACH_COORDINATES[matching_beach]['lon']],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(color=color),
                        tooltip=beach_name
                    ).add_to(m)
            
            # Display the map
            st_folium(m, width=800, height=400)
            
            # Add legend
            st.markdown("""
            ### Map Legend
            - 🟢 Green markers: Water quality within safe limits
            - 🔴 Red markers: Water quality exceeds safe limits
            """)
            
            # Beach Comparison Chart Section
            st.markdown("""
                <div style='margin-top: 16px; margin-bottom: 24px;'>
                    <h2 style='font-size: 18px; color: #333333; font-weight: bold;'>Beach Comparison Over Time</h2>
                    <p style='font-size: 14px; color: #666666;'>Track bacteria levels across NYC beaches</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create filter section with responsive layout
            if st.session_state.get('is_mobile', False):
                period = st.selectbox(
                    "Select Time Period",
                    options=["Summer 2025", "5 Years", "10 Years"],
                    index=0,
                    key="beach_comparison_period"
                )
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    period = st.selectbox(
                        "Select Time Period",
                        options=["Summer 2025", "5 Years", "10 Years"],
                        index=0,
                        key="beach_comparison_period"
                    )
            
            # Define major beaches and their colors
            major_beaches = {
                'Coney Island': 'green',
                'Kingsborough Community College': 'blue',
                'Manhattan Beach': 'orange',
                'Breezy Point': 'purple',
                'Rockaway Beach': 'red',
                'Orchard Beach': 'brown',
                'Midland Beach': 'pink',
                'South Beach': 'cyan',
                'Cedar Grove Beach': 'magenta'
            }
            
            # Filter data based on selected period
            if period == 'Summer 2025':
                start_date = pd.Timestamp('2025-06-01')
                end_date = pd.Timestamp('2025-08-31')
            elif period == '5 Years':
                start_date = pd.Timestamp('2020-01-01')
                end_date = pd.Timestamp('2025-12-31')
            else:  # 10 Years
                start_date = pd.Timestamp('2015-01-01')
                end_date = pd.Timestamp('2025-12-31')
            
            # Filter data for selected beaches and time period
            mask = (df['Beach Name'].isin(major_beaches.keys())) & \
                   (df['Sample Date'] >= start_date) & \
                   (df['Sample Date'] <= end_date)
            filtered_df = df[mask]
            
            # Create the line chart
            fig = go.Figure()
            
            # Add a line for each beach
            for beach, color in major_beaches.items():
                beach_data = filtered_df[filtered_df['Beach Name'] == beach]
                if not beach_data.empty:
                    # Calculate trend
                    trend = beach_data['Enterococci Results'].iloc[-1] - beach_data['Enterococci Results'].iloc[0]
                    trend_symbol = '↗️' if trend > 0 else '↘️'
                    
                    # Sort data by date to ensure proper line connection
                    beach_data = beach_data.sort_values('Sample Date')
                    
                    fig.add_trace(go.Scatter(
                        x=beach_data['Sample Date'],
                        y=beach_data['Enterococci Results'],
                        name=f"{beach} {trend_symbol}",
                        line=dict(color=color, width=2),
                        hovertemplate="Date: %{x}<br>Level: %{y:.1f} MPN/100ml<extra></extra>"
                    ))
            
            # Add EPA safety threshold line
            fig.add_hline(
                y=safe_limit,
                line_dash="dash",
                line_color="red",
                annotation_text="EPA Safety Threshold",
                annotation_position="right"
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "Beach Comparison Over Time",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, color='#333333')
                },
                xaxis_title="Date",
                yaxis_title="Bacteria Level (MPN/100ml)",
                yaxis=dict(range=[0, max(filtered_df['Enterococci Results'].max() * 1.1, safe_limit * 1.1)]),
                height=350 if st.session_state.get('is_mobile', False) else 280,
                margin=dict(l=16, r=16, t=60, b=60),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                hovermode="x unified",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            
            # Display the chart with loading spinner
            with st.spinner("Loading beach comparison data..."):
                st.plotly_chart(fig, use_container_width=True)
                
            # Add 2024 Comparison Chart
            st.markdown("""
                <div style='margin-top: 32px; margin-bottom: 24px;'>
                    <h2 style='font-size: 18px; color: #333333; font-weight: bold;'>2024 Beach Comparison</h2>
                    <p style='font-size: 14px; color: #666666;'>Track bacteria levels across NYC beaches in 2024</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Filter data for 2024
            mask_2024 = (df['Beach Name'].isin(major_beaches.keys())) & \
                       (df['Sample Date'] >= pd.Timestamp('2024-01-01')) & \
                       (df['Sample Date'] <= pd.Timestamp('2024-12-31'))
            filtered_df_2024 = df[mask_2024]
            
            # Create the 2024 line chart
            fig_2024 = go.Figure()
            
            # Add a line for each beach
            for beach, color in major_beaches.items():
                beach_data = filtered_df_2024[filtered_df_2024['Beach Name'] == beach]
                if not beach_data.empty:
                    # Calculate trend
                    trend = beach_data['Enterococci Results'].iloc[-1] - beach_data['Enterococci Results'].iloc[0]
                    trend_symbol = '↗️' if trend > 0 else '↘️'
                    
                    # Sort data by date to ensure proper line connection
                    beach_data = beach_data.sort_values('Sample Date')
                    
                    fig_2024.add_trace(go.Scatter(
                        x=beach_data['Sample Date'],
                        y=beach_data['Enterococci Results'],
                        name=f"{beach} {trend_symbol}",
                        line=dict(color=color, width=2),
                        hovertemplate="Date: %{x}<br>Level: %{y:.1f} MPN/100ml<extra></extra>"
                    ))
            
            # Add EPA safety threshold line
            fig_2024.add_hline(
                y=35,  # EPA safety threshold
                line_dash="dash",
                line_color="red",
                annotation_text="EPA Safety Threshold",
                annotation_position="right"
            )
            
            # Update layout for 2024 chart
            fig_2024.update_layout(
                title={
                    'text': "2024 Beach Comparison",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, color='#333333')
                },
                xaxis_title="Date",
                yaxis_title="Bacteria Level (MPN/100ml)",
                yaxis=dict(range=[0, 100]),  # Fixed scale as requested
                height=350 if st.session_state.get('is_mobile', False) else 280,
                margin=dict(l=16, r=16, t=60, b=60),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                hovermode="x unified",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            
            # Display the 2024 chart with loading spinner
            with st.spinner("Loading 2024 beach comparison data..."):
                st.plotly_chart(fig_2024, use_container_width=True)
                
            # Add debug information for 2024 data
            st.write("2024 Data Summary:")
            st.write("Number of records in 2024:", len(filtered_df_2024))
            st.write("2024 data preview:")
            st.dataframe(filtered_df_2024[['Beach Name', 'Sample Date', 'Enterococci Results']].head())
        
        with chat_col:
            st.markdown("### Beach Water Quality Helper")
            st.markdown("""
            Hi! I'm your friendly beach water quality helper! I can explain things in simple terms and help you understand 
            what makes our beaches safe and healthy. Click on a beach marker on the map and ask me anything!
            """)
            
            # Initialize chat history
            if "map_comparison_chat_messages" not in st.session_state:
                st.session_state.map_comparison_chat_messages = []
            
            # Display chat messages from history
            for message in st.session_state.map_comparison_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            if prompt := st.chat_input("Ask me about beach water quality..."):
                # Add user message to chat history
                st.session_state.map_comparison_chat_messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Prepare context from the dataset
                context = f"""
                You are a friendly beach water quality helper who loves teaching kids about water safety! 
                Keep your answers short and sweet - like a quick chat with a friend.

                Quick facts about our beaches:
                • {len(df)} water checks done
                • Checking since {df['Sample Date'].min().strftime('%Y-%m-%d')}
                • Beaches: {', '.join(df['Beach Name'].unique())}
                • Safe level: {safe_limit} MPN/100ml

                Latest checks:
                {df.sort_values('Sample Date').groupby('Beach Name').last()[['Sample Date', 'Enterococci Results']].to_string()}

                Fun facts:
                • {len(df[df['Enterococci Results'] > safe_limit])} times water wasn't clean enough
                • Average: {df['Enterococci Results'].mean():.2f} MPN/100ml
                • Highest: {df['Enterococci Results'].max():.2f} MPN/100ml

                Tips for talking to kids:
                • Keep it short and fun
                • Use simple words
                • Make it like telling a quick story
                • Compare numbers to things they know
                • Use our real data
                """
                
                # Prepare the full prompt with context
                full_prompt = f"""Context: {context}

                User question: {prompt}

                Remember: Keep your answer short and sweet! Use our data to give accurate answers, 
                but explain everything in a fun, simple way that a kid can understand quickly."""
                
                # Display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Prepare API request
                    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                    headers = {
                        "Authorization": "Bearer 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",
                        "Accept": "text/event-stream"
                    }
                    
                    payload = {
                        "model": "meta/llama-4-scout-17b-16e-instruct",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 1.00,
                        "frequency_penalty": 0.00,
                        "presence_penalty": 0.00,
                        "stream": True
                    }
                    
                    try:
                        response = requests.post(invoke_url, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode("utf-8").replace("data: ", ""))
                                    if "choices" in data and len(data["choices"]) > 0:
                                        content = data["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            full_response += content
                                            message_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    continue
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.map_comparison_chat_messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Error communicating with AI service: {str(e)}")
                        st.info("Please try again later or rephrase your question.")
    elif view == "Beach Comparison Test 1":
        st.header("Beach Comparison Over Time")
        st.markdown("Track bacteria levels across NYC beaches")
        
        # Create filter section with responsive layout
        col1, col2, col3 = st.columns(3)
        with col1:
            summer_2025 = st.selectbox(
                "Summer 2025",
                options=["Jun 2025", "Jul 2025", "Aug 2025"],
                index=0,
                key="summer_2025"
            )
        with col2:
            five_years = st.selectbox(
                "5 Years",
                options=["2020", "2021", "2022", "2023", "2024", "2025"],
                index=5,
                key="five_years"
            )
        with col3:
            ten_years = st.selectbox(
                "10 Years",
                options=["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
                index=10,
                key="ten_years"
            )
        
        # Debug: Show available beach names in dataset
        st.write("Available beaches in dataset:", sorted(df['Beach Name'].unique()))
        
        # Define major beaches and their colors
        major_beaches = {
            'MIDLAND BEACH': 'green',
            'MANHATTAN BEACH': 'blue',
            'SOUTH BEACH': 'orange',
            'CONEY ISLAND': 'purple',
            'ROCKAWAY BEACH': 'red'
        }
        
        # Filter data based on selected period
        if summer_2025:
            start_date = pd.Timestamp(f'2025-{summer_2025.split()[0]}-01')
            end_date = pd.Timestamp(f'2025-{summer_2025.split()[0]}-30')
        elif five_years:
            start_date = pd.Timestamp(f'{five_years}-01-01')
            end_date = pd.Timestamp(f'{five_years}-12-31')
        else:  # ten_years
            start_date = pd.Timestamp(f'{ten_years}-01-01')
            end_date = pd.Timestamp(f'{ten_years}-12-31')
        
        # Debug: Show date range
        st.write("Selected date range:", start_date, "to", end_date)
        
        # Filter data for selected beaches and time period
        mask = (df['Beach Name'].isin(major_beaches.keys())) & \
               (df['Sample Date'] >= start_date) & \
               (df['Sample Date'] <= end_date)
        filtered_df = df[mask]
        
        # Debug: Show filtered data
        st.write("Number of records after filtering:", len(filtered_df))
        st.write("Filtered data preview:")
        st.dataframe(filtered_df[['Beach Name', 'Sample Date', 'Enterococci Results']].head())
        
        # Create the line chart
        fig = go.Figure()
        
        # Add a line for each beach
        for beach, color in major_beaches.items():
            beach_data = filtered_df[filtered_df['Beach Name'] == beach]
            if not beach_data.empty:
                # Calculate trend
                trend = beach_data['Enterococci Results'].iloc[-1] - beach_data['Enterococci Results'].iloc[0]
                trend_symbol = '↗️' if trend > 0 else '↘️'
                
                # Sort data by date to ensure proper line connection
                beach_data = beach_data.sort_values('Sample Date')
                
                # Debug: Show data for each beach
                st.write(f"Data points for {beach}:", len(beach_data))
                
                fig.add_trace(go.Scatter(
                    x=beach_data['Sample Date'],
                    y=beach_data['Enterococci Results'],
                    name=f"{beach} {trend_symbol}",
                    line=dict(color=color, width=2),
                    hovertemplate="Date: %{x}<br>Level: %{y:.1f} MPN/100ml<extra></extra>"
                ))
        
        # Add EPA safety threshold line
        fig.add_hline(
            y=35,  # EPA safety threshold
            line_dash="dash",
            line_color="red",
            annotation_text="EPA Safety Threshold",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Beach Comparison Over Time",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, color='#333333')
            },
            xaxis_title="Date",
            yaxis_title="Bacteria Level (MPN/100ml)",
            yaxis=dict(range=[0, 100]),  # Fixed scale as requested
            height=350 if st.session_state.get('is_mobile', False) else 280,
            margin=dict(l=16, r=16, t=60, b=60),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        # Display the chart with loading spinner
        with st.spinner("Loading beach comparison data..."):
            st.plotly_chart(fig, use_container_width=True)
            
        # Add 2024 Comparison Chart
        st.markdown("""
            <div style='margin-top: 32px; margin-bottom: 24px;'>
                <h2 style='font-size: 18px; color: #333333; font-weight: bold;'>2024 Beach Comparison</h2>
                <p style='font-size: 14px; color: #666666;'>Track bacteria levels across NYC beaches in 2024</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Filter data for 2024
        mask_2024 = (df['Beach Name'].isin(major_beaches.keys())) & \
                   (df['Sample Date'] >= pd.Timestamp('2024-01-01')) & \
                   (df['Sample Date'] <= pd.Timestamp('2024-12-31'))
        filtered_df_2024 = df[mask_2024]
        
        # Create the 2024 line chart
        fig_2024 = go.Figure()
        
        # Add a line for each beach
        for beach, color in major_beaches.items():
            beach_data = filtered_df_2024[filtered_df_2024['Beach Name'] == beach]
            if not beach_data.empty:
                # Calculate trend
                trend = beach_data['Enterococci Results'].iloc[-1] - beach_data['Enterococci Results'].iloc[0]
                trend_symbol = '↗️' if trend > 0 else '↘️'
                
                # Sort data by date to ensure proper line connection
                beach_data = beach_data.sort_values('Sample Date')
                
                fig_2024.add_trace(go.Scatter(
                    x=beach_data['Sample Date'],
                    y=beach_data['Enterococci Results'],
                    name=f"{beach} {trend_symbol}",
                    line=dict(color=color, width=2),
                    hovertemplate="Date: %{x}<br>Level: %{y:.1f} MPN/100ml<extra></extra>"
                ))
        
        # Add EPA safety threshold line
        fig_2024.add_hline(
            y=35,  # EPA safety threshold
            line_dash="dash",
            line_color="red",
            annotation_text="EPA Safety Threshold",
            annotation_position="right"
        )
        
        # Update layout for 2024 chart
        fig_2024.update_layout(
            title={
                'text': "2024 Beach Comparison",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, color='#333333')
            },
            xaxis_title="Date",
            yaxis_title="Bacteria Level (MPN/100ml)",
            yaxis=dict(range=[0, 100]),  # Fixed scale as requested
            height=350 if st.session_state.get('is_mobile', False) else 280,
            margin=dict(l=16, r=16, t=60, b=60),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        # Display the 2024 chart with loading spinner
        with st.spinner("Loading 2024 beach comparison data..."):
            st.plotly_chart(fig_2024, use_container_width=True)
            
        # Add debug information for 2024 data
        st.write("2024 Data Summary:")
        st.write("Number of records in 2024:", len(filtered_df_2024))
        st.write("2024 data preview:")
        st.dataframe(filtered_df_2024[['Beach Name', 'Sample Date', 'Enterococci Results']].head())
else:
    st.error("Error loading data. Please check the console for more information.") 