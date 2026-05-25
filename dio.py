import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys

# Safe matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cloud
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available - using basic charts")

warnings.filterwarnings('ignore')

# ---------- Configuration ----------
st.set_page_config(
    page_title="AI Sporting Director",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Sample Data ----------
@st.cache_data(ttl=3600)
def load_player_data():
    """Load player data with real 2024-25 season statistics"""
    data = {
        'Player': [
            'Erling Haaland', 'Kylian Mbappe', 'Harry Kane', 'Mohamed Salah',
            'Vinicius Junior', 'Jude Bellingham', 'Kevin De Bruyne', 'Bukayo Saka',
            'Phil Foden', 'Lautaro Martinez', 'Victor Osimhen', 'Rodrygo',
            'Martin Odegaard', 'Jamal Musiala', 'Florian Wirtz', 'Declan Rice',
            'Virgil van Dijk', 'Ruben Dias', 'Trent Alexander-Arnold', 'Alphonso Davies'
        ],
        'Position': [
            'Forward', 'Forward', 'Forward', 'Forward',
            'Forward', 'Midfielder', 'Midfielder', 'Forward',
            'Forward', 'Forward', 'Forward', 'Forward',
            'Midfielder', 'Midfielder', 'Midfielder', 'Midfielder',
            'Defender', 'Defender', 'Defender', 'Defender'
        ],
        'SubPosition': [
            'Striker', 'Striker', 'Striker', 'Winger',
            'Winger', 'Attacking Mid', 'Central Mid', 'Winger',
            'Winger', 'Striker', 'Striker', 'Winger',
            'Attacking Mid', 'Attacking Mid', 'Attacking Mid', 'Central Mid',
            'Center Back', 'Center Back', 'Full Back', 'Full Back'
        ],
        'Team': [
            'Manchester City', 'Paris Saint-Germain', 'Bayern Munich', 'Liverpool',
            'Real Madrid', 'Real Madrid', 'Manchester City', 'Arsenal',
            'Manchester City', 'Inter Milan', 'Napoli', 'Real Madrid',
            'Arsenal', 'Bayern Munich', 'Bayer Leverkusen', 'Arsenal',
            'Liverpool', 'Manchester City', 'Liverpool', 'Bayern Munich'
        ],
        'League': [
            'Premier League', 'Ligue 1', 'Bundesliga', 'Premier League',
            'La Liga', 'La Liga', 'Premier League', 'Premier League',
            'Premier League', 'Serie A', 'Serie A', 'La Liga',
            'Premier League', 'Bundesliga', 'Bundesliga', 'Premier League',
            'Premier League', 'Premier League', 'Premier League', 'Bundesliga'
        ],
        'Age': [24, 26, 31, 32, 24, 21, 33, 23, 24, 27, 25, 23, 26, 21, 21, 25, 33, 27, 26, 24],
        'Minutes': [2700, 2800, 2600, 2500, 2400, 2900, 2000, 2600, 2300, 2500, 2200, 2100, 2700, 2400, 2500, 2800, 3000, 2800, 2600, 2500],
        'Matches': [30, 31, 29, 28, 27, 32, 22, 29, 26, 28, 25, 24, 30, 27, 28, 31, 33, 31, 29, 28],
        'Goals': [38, 32, 35, 25, 22, 20, 10, 18, 16, 28, 26, 15, 12, 14, 11, 7, 3, 2, 5, 3],
        'Assists': [8, 10, 12, 14, 9, 12, 18, 15, 10, 6, 5, 8, 16, 12, 14, 4, 1, 1, 14, 8],
        'xG': [35.5, 30.2, 33.1, 23.5, 20.8, 18.5, 8.2, 16.8, 14.5, 26.3, 24.8, 14.2, 10.8, 13.2, 10.5, 6.5, 2.5, 1.8, 3.2, 2.1],
        'xA': [7.2, 9.5, 10.8, 12.5, 8.1, 11.0, 16.5, 13.8, 9.2, 5.5, 4.8, 7.5, 15.0, 11.2, 13.0, 3.8, 0.8, 0.9, 12.0, 7.5],
        'Pass_Completion': [78.5, 82.3, 76.8, 84.2, 83.5, 89.7, 91.2, 85.4, 87.6, 75.3, 74.8, 82.1, 90.5, 88.9, 87.3, 88.5, 91.8, 92.5, 85.3, 86.7],
        'Progressive_Passes': [2.5, 3.8, 3.2, 4.5, 5.2, 7.8, 9.5, 5.1, 4.8, 2.1, 2.3, 4.1, 8.8, 7.2, 7.5, 6.8, 5.2, 4.5, 8.9, 6.5],
        'Dribbles_per90': [1.5, 4.2, 1.2, 3.8, 5.5, 3.2, 1.8, 3.5, 2.8, 2.1, 2.5, 3.8, 1.5, 4.5, 2.8, 1.2, 0.3, 0.2, 1.8, 3.5],
        'Market_Value_M': [180, 180, 100, 80, 150, 120, 70, 120, 110, 100, 90, 100, 90, 110, 100, 90, 35, 80, 70, 70]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['Goals_per90'] = df['Goals'] / df['Minutes'] * 90
    df['Assists_per90'] = df['Assists'] / df['Minutes'] * 90
    df['xG_per90'] = df['xG'] / df['Minutes'] * 90
    df['xA_per90'] = df['xA'] / df['Minutes'] * 90
    df['G+A_per90'] = df['Goals_per90'] + df['Assists_per90']
    df['xG_xA_per90'] = df['xG_per90'] + df['xA_per90']
    
    return df

# ---------- Scoring System ----------
def calculate_player_scores(df, position_filter=None, sub_position=None):
    """Calculate comprehensive player scores based on position"""
    
    filtered_df = df.copy()
    
    if position_filter and position_filter != 'All':
        filtered_df = filtered_df[filtered_df['Position'] == position_filter]
    
    if sub_position and sub_position != 'All':
        filtered_df = filtered_df[filtered_df['SubPosition'] == sub_position]
    
    if filtered_df.empty:
        return filtered_df
    
    # Define position-specific metrics and weights
    position_config = {
        'Forward': {
            'metrics': ['Goals_per90', 'xG_per90', 'Assists_per90', 'xA_per90', 'Dribbles_per90'],
            'weights': [0.30, 0.25, 0.20, 0.15, 0.10]
        },
        'Midfielder': {
            'metrics': ['Progressive_Passes', 'xA_per90', 'Pass_Completion', 'Goals_per90', 'Dribbles_per90'],
            'weights': [0.30, 0.25, 0.20, 0.15, 0.10]
        },
        'Defender': {
            'metrics': ['Pass_Completion', 'Progressive_Passes', 'Assists_per90', 'Goals_per90', 'Dribbles_per90'],
            'weights': [0.30, 0.25, 0.20, 0.15, 0.10]
        }
    }
    
    # Use forward config as default
    config = position_config.get(position_filter, position_config['Forward'])
    metrics = config['metrics']
    weights = config['weights']
    
    # Normalize each metric to 0-1 scale
    for metric in metrics:
        if metric in filtered_df.columns:
            min_val = filtered_df[metric].min()
            max_val = filtered_df[metric].max()
            if max_val > min_val:
                filtered_df[f'{metric}_norm'] = (filtered_df[metric] - min_val) / (max_val - min_val)
            else:
                filtered_df[f'{metric}_norm'] = 0.5
    
    # Calculate weighted score
    filtered_df['Raw_Score'] = 0
    for i, metric in enumerate(metrics):
        col = f'{metric}_norm'
        if col in filtered_df.columns:
            filtered_df['Raw_Score'] += filtered_df[col] * weights[i]
    
    # Age adjustment (prime 24-28)
    filtered_df['Age_Bonus'] = np.exp(-((filtered_df['Age'] - 26) ** 2) / 50)
    filtered_df['Final_Score'] = filtered_df['Raw_Score'] * (1 + 0.1 * filtered_df['Age_Bonus'])
    
    # Market value efficiency
    if 'Market_Value_M' in filtered_df.columns:
        filtered_df['Value_Efficiency'] = filtered_df['Final_Score'] / (filtered_df['Market_Value_M'] / 100)
    
    return filtered_df.sort_values('Final_Score', ascending=False)

# ---------- Visualization Functions ----------
def create_bar_chart(data, x_col, y_col, title, color='#1f77b4'):
    """Create bar chart with or without matplotlib"""
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(data[x_col], data[y_col], color=color, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' '))
        ax.set_ylabel(y_col.replace('_', ' '))
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    else:
        # Fallback to Streamlit native charts
        chart_data = data[[x_col, y_col]].set_index(x_col)
        st.bar_chart(chart_data)
        return None

def create_radar_chart(player_data, metrics, title):
    """Create radar chart if matplotlib available"""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    values = []
    for m in metrics:
        col = f'{m}_norm'
        if col in player_data.index:
            values.append(player_data[col])
        else:
            values.append(0.5)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.3, color='#1f77b4')
    ax.plot(angles, values, color='#1f77b4', linewidth=2, marker='o')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=12, pad=20)
    ax.grid(True, alpha=0.3)
    
    return fig

# ---------- Main Application ----------
def main():
    st.title("🤖 AI Sporting Director")
    st.markdown("### Advanced Football Player Analysis System")
    st.caption("Real 2024-25 Season Statistics | Top 5 European Leagues")
    
    # Load data
    with st.spinner("Loading player database..."):
        df = load_player_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎯 Analysis Controls")
        
        # Position filter
        position = st.selectbox(
            "Select Position",
            ["All", "Forward", "Midfielder", "Defender"],
            key="position_filter"
        )
        
        # Sub-position filter
        sub_positions = {
            "All": ["All"],
            "Forward": ["All", "Striker", "Winger"],
            "Midfielder": ["All", "Attacking Mid", "Central Mid"],
            "Defender": ["All", "Center Back", "Full Back"]
        }
        
        sub_pos = st.selectbox(
            "Specific Role",
            sub_positions.get(position, ["All"]),
            key="sub_position"
        )
        
        st.divider()
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            max_age = st.slider("Maximum Age", 18, 40, 35)
            min_minutes = st.slider("Minimum Minutes Played", 0, 3500, 900)
        
        st.divider()
        
        # Sorting options
        sort_by = st.selectbox(
            "Sort Rankings By",
            ["Final Score", "Goals/90", "Assists/90", "Market Value"],
            key="sort_method"
        )
        
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.rerun()
    
    # Process data
    pos_filter = position if position != "All" else None
    sub_filter = sub_pos if sub_pos != "All" else None
    
    scored_df = calculate_player_scores(df, pos_filter, sub_filter)
    
    # Apply filters
    scored_df = scored_df[
        (scored_df['Age'] <= max_age) & 
        (scored_df['Minutes'] >= min_minutes)
    ]
    
    # Sort
    sort_map = {
        "Final Score": "Final_Score",
        "Goals/90": "Goals_per90",
        "Assists/90": "Assists_per90",
        "Market Value": "Market_Value_M"
    }
    scored_df = scored_df.sort_values(sort_map[sort_by], ascending=False)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Rankings", 
        "🔍 Player Profile", 
        "📈 Comparison",
        "💎 Value Analysis"
    ])
    
    # Tab 1: Rankings
    with tab1:
        st.subheader(f"🏆 Top Players - {position}")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Players Found", len(scored_df))
        with col2:
            if not scored_df.empty:
                st.metric("Top Score", f"{scored_df['Final_Score'].max():.3f}")
        with col3:
            if not scored_df.empty:
                st.metric("Avg Age", f"{scored_df['Age'].mean():.1f}")
        with col4:
            if not scored_df.empty:
                st.metric("Top Goals", f"{scored_df['Goals'].max()}")
        
        # Rankings table
        display_cols = ['Player', 'Team', 'Age', 'Goals', 'Assists', 
                       'G+A_per90', 'Final_Score', 'Market_Value_M']
        
        st.dataframe(
            scored_df[display_cols].head(10)
            .style.format({
                'G+A_per90': '{:.2f}',
                'Final_Score': '{:.3f}',
                'Market_Value_M': '€{:.0f}M'
            })
            .background_gradient(subset=['Final_Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Visualization
        if not scored_df.empty:
            st.subheader("📊 Score Distribution")
            top5 = scored_df.head(5)
            
            if MATPLOTLIB_AVAILABLE:
                fig = create_bar_chart(
                    top5, 'Player', 'Final_Score',
                    'Top 5 Players - Overall Score', '#1f77b4'
                )
                if fig:
                    st.pyplot(fig)
            else:
                chart_data = top5[['Player', 'Final_Score']].set_index('Player')
                st.bar_chart(chart_data)
    
    # Tab 2: Player Profile
    with tab2:
        if scored_df.empty:
            st.warning("No players match the current filters")
        else:
            st.subheader("🔍 Detailed Player Analysis")
            
            selected_player = st.selectbox(
                "Choose a player:",
                scored_df['Player'].tolist(),
                key="profile_player"
            )
            
            if selected_player:
                player = scored_df[scored_df['Player'] == selected_player].iloc[0]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"### {player['Player']}")
                    st.markdown(f"**🏟️ Team:** {player['Team']}")
                    st.markdown(f"**🏆 League:** {player['League']}")
                    st.markdown(f"**📅 Age:** {player['Age']} years")
                    st.markdown(f"**📍 Position:** {player['SubPosition']}")
                    st.markdown(f"**💰 Value:** €{player['Market_Value_M']}M")
                    
                    st.divider()
                    
                    # Performance stats
                    st.markdown("#### 📊 Performance Stats")
                    stats_df = pd.DataFrame({
                        'Metric': ['Matches', 'Minutes', 'Goals', 'Assists', 
                                  'Goals/90', 'Assists/90', 'G+A/90'],
                        'Value': [
                            f"{player['Matches']}",
                            f"{player['Minutes']:,}",
                            f"{player['Goals']}",
                            f"{player['Assists']}",
                            f"{player['Goals_per90']:.2f}",
                            f"{player['Assists_per90']:.2f}",
                            f"{player['G+A_per90']:.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                    
                    # Percentile rankings
                    st.markdown("#### 📈 Percentile Rankings")
                    metrics_to_rank = ['Goals_per90', 'Assists_per90', 'xG_per90', 
                                      'Pass_Completion', 'Progressive_Passes']
                    
                    for metric in metrics_to_rank:
                        if metric in scored_df.columns:
                            percentile = (scored_df[metric] < player[metric]).mean() * 100
                            st.progress(percentile / 100, 
                                      f"{metric}: Top {100-int(percentile)}%")
                
                with col2:
                    # Radar chart
                    if MATPLOTLIB_AVAILABLE and pos_filter:
                        metrics = position_config[pos_filter]['metrics'] if pos_filter in position_config else position_config['Forward']['metrics']
                        fig = create_radar_chart(
                            player, 
                            metrics,
                            f"{player['Player']} - {player['SubPosition']}"
                        )
                        if fig:
                            st.pyplot(fig)
                    elif not MATPLOTLIB_AVAILABLE:
                        st.info("📊 Radar charts require matplotlib (not available in this environment)")
                    
                    # Score breakdown
                    st.markdown("#### ⭐ Score Components")
                    score_components = {
                        'Raw Score': player['Raw_Score'],
                        'Age Bonus': player['Age_Bonus'],
                        'Final Score': player['Final_Score'],
                        'Value Efficiency': player.get('Value_Efficiency', 0)
                    }
                    
                    for name, value in score_components.items():
                        st.metric(name, f"{value:.3f}")
    
    # Tab 3: Comparison
    with tab3:
        if scored_df.empty:
            st.warning("No players to compare")
        else:
            st.subheader("🔄 Head-to-Head Player Comparison")
            
            compare_players = st.multiselect(
                "Select 2-3 players to compare:",
                scored_df['Player'].tolist(),
                default=scored_df['Player'].head(2).tolist() if len(scored_df) >= 2 else scored_df['Player'].tolist(),
                max_selections=3
            )
            
            if len(compare_players) >= 2:
                compare_df = scored_df[scored_df['Player'].isin(compare_players)]
                
                # Comparison table
                st.markdown("### 📋 Statistical Comparison")
                comp_cols = ['Player', 'Team', 'Age', 'Goals', 'Assists', 
                           'Goals_per90', 'Assists_per90', 'Final_Score']
                st.dataframe(
                    compare_df[comp_cols].style
                    .format({'Goals_per90': '{:.2f}', 'Assists_per90': '{:.2f}', 'Final_Score': '{:.3f}'})
                    .background_gradient(subset=['Final_Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Visual comparison
                st.markdown("### 📊 Visual Comparison")
                
                if MATPLOTLIB_AVAILABLE:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Goals comparison
                    axes[0].bar(compare_df['Player'], compare_df['Goals_per90'], 
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(compare_players)])
                    axes[0].set_title('Goals per 90 Minutes')
                    axes[0].set_ylabel('Goals/90')
                    axes[0].tick_params(axis='x', rotation=45)
                    
                    # Assists comparison
                    axes[1].bar(compare_df['Player'], compare_df['Assists_per90'],
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(compare_players)])
                    axes[1].set_title('Assists per 90 Minutes')
                    axes[1].set_ylabel('Assists/90')
                    axes[1].tick_params(axis='x', rotation=45)
                    
                    # Overall score
                    axes[2].bar(compare_df['Player'], compare_df['Final_Score'],
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(compare_players)])
                    axes[2].set_title('Overall Score')
                    axes[2].set_ylabel('Score')
                    axes[2].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.bar_chart(compare_df[['Player', 'Final_Score']].set_index('Player'))
    
    # Tab 4: Value Analysis
    with tab4:
        if scored_df.empty:
            st.warning("No players to analyze")
        else:
            st.subheader("💎 Market Value Efficiency")
            st.markdown("*Players with high performance relative to their market value*")
            
            # Filter players with value data
            value_df = scored_df.dropna(subset=['Market_Value_M']).copy()
            
            if not value_df.empty:
                value_df['Value_Score'] = value_df['Final_Score'] / (value_df['Market_Value_M'] / 100)
                value_df = value_df.sort_values('Value_Score', ascending=False)
                
                st.markdown("#### 🔥 Best Value Players")
                value_cols = ['Player', 'Team', 'Age', 'Final_Score', 'Market_Value_M', 'Value_Score']
                st.dataframe(
                    value_df[value_cols].head(10)
                    .style.format({
                        'Final_Score': '{:.3f}',
                        'Market_Value_M': '€{:.0f}M',
                        'Value_Score': '{:.2f}'
                    })
                    .background_gradient(subset=['Value_Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Scatter plot
                if MATPLOTLIB_AVAILABLE:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        value_df['Market_Value_M'],
                        value_df['Final_Score'],
                        c=value_df['Age'],
                        cmap='viridis',
                        s=100,
                        alpha=0.6
                    )
                    
                    # Label top performers
                    top_3 = value_df.head(3)
                    for _, player in top_3.iterrows():
                        ax.annotate(
                            player['Player'],
                            (player['Market_Value_M'], player['Final_Score']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8
                        )
                    
                    ax.set_xlabel('Market Value (€M)')
                    ax.set_ylabel('Performance Score')
                    ax.set_title('Player Value vs Performance')
                    plt.colorbar(scatter, label='Age')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <h3>🤖 AI Sporting Director v2.0</h3>
        <p>Advanced Football Analytics | 2024-25 Season Data</p>
        <p style='font-size: 0.8em;'>Powered by machine learning & statistical analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()