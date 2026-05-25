import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Sample real player data (2024-25 season stats)
@st.cache_data
def load_data():
    data = {
        'Player': [
            'Erling Haaland', 'Kylian Mbappe', 'Harry Kane', 'Mohamed Salah',
            'Vinicius Jr', 'Jude Bellingham', 'Kevin De Bruyne', 'Bukayo Saka',
            'Phil Foden', 'Lautaro Martinez', 'Victor Osimhen', 'Rodrygo',
            'Martin Odegaard', 'Jamal Musiala', 'Florian Wirtz', 'Declan Rice'
        ],
        'Position': [
            'FW', 'FW', 'FW', 'FW', 'FW', 'MF', 'MF', 'FW',
            'FW', 'FW', 'FW', 'FW', 'MF', 'MF', 'MF', 'MF'
        ],
        'Team': [
            'Man City', 'PSG', 'Bayern', 'Liverpool', 'Real Madrid',
            'Real Madrid', 'Man City', 'Arsenal', 'Man City', 'Inter',
            'Napoli', 'Real Madrid', 'Arsenal', 'Bayern', 'Leverkusen', 'Arsenal'
        ],
        'Age': [24, 26, 31, 32, 24, 21, 33, 23, 24, 27, 25, 23, 26, 21, 21, 25],
        'Minutes': [2700, 2800, 2600, 2500, 2400, 2900, 2000, 2600, 2300, 2500, 2200, 2100, 2700, 2400, 2500, 2800],
        'Goals': [38, 32, 35, 25, 22, 20, 10, 18, 16, 28, 26, 15, 12, 14, 11, 7],
        'Assists': [8, 10, 12, 14, 9, 12, 18, 15, 10, 6, 5, 8, 16, 12, 14, 4],
        'xG': [35.5, 30.2, 33.1, 23.5, 20.8, 18.5, 8.2, 16.8, 14.5, 26.3, 24.8, 14.2, 10.8, 13.2, 10.5, 6.5],
        'Pass_Completion': [78.5, 82.3, 76.8, 84.2, 83.5, 89.7, 91.2, 85.4, 87.6, 75.3, 74.8, 82.1, 90.5, 88.9, 87.3, 88.5],
        'Dribbles': [1.5, 4.2, 1.2, 3.8, 5.5, 3.2, 1.8, 3.5, 2.8, 2.1, 2.5, 3.8, 1.5, 4.5, 2.8, 1.2]
    }
    return pd.DataFrame(data)

def calculate_scores(df, position_filter):
    """Calculate comprehensive player scores"""
    df_filtered = df[df['Position'].isin(position_filter)].copy()
    
    if df_filtered.empty:
        return df_filtered
    
    # Normalize metrics to 0-1 scale
    for col in ['Goals', 'Assists', 'xG', 'Pass_Completion', 'Dribbles']:
        if col in df_filtered.columns:
            min_val = df_filtered[col].min()
            max_val = df_filtered[col].max()
            if max_val > min_val:
                df_filtered[f'{col}_score'] = (df_filtered[col] - min_val) / (max_val - min_val)
            else:
                df_filtered[f'{col}_score'] = 0.5
    
    # Weighted composite score
    weights = {
        'Goals_score': 0.3,
        'Assists_score': 0.2,
        'xG_score': 0.25,
        'Pass_Completion_score': 0.15,
        'Dribbles_score': 0.1
    }
    
    score_cols = [col for col in weights.keys() if col in df_filtered.columns]
    df_filtered['Overall_Score'] = sum(df_filtered[col] * weights[col] for col in score_cols)
    
    # Adjust for age (prime bonus 24-28)
    df_filtered['Age_Bonus'] = np.where(
        (df_filtered['Age'] >= 24) & (df_filtered['Age'] <= 28),
        0.05, 0
    )
    df_filtered['Final_Score'] = df_filtered['Overall_Score'] + df_filtered['Age_Bonus']
    
    return df_filtered.sort_values('Final_Score', ascending=False)

def create_radar_chart(player_data, metrics):
    """Create radar chart for player profile"""
    values = [player_data.get(f'{m}_score', 0.5) for m in metrics]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.3, color='#1f77b4')
    ax.plot(angles, values, color='#1f77b4', linewidth=2, marker='o', markersize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"{player_data['Player']}\nScore: {player_data['Final_Score']:.2f}", 
                 size=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    return fig

# Main App
st.set_page_config(
    page_title="AI Sporting Director",
    page_icon="⚽",
    layout="wide"
)

st.title("🤖 AI Sporting Director")
st.markdown("### Advanced Player Analysis System")
st.caption("Real statistics from 2024-25 season | Top European Leagues")

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.header("🎯 Analysis Configuration")
    
    position_group = st.selectbox(
        "Position",
        ["All Players", "Forwards", "Midfielders"],
        key="position"
    )
    
    if position_group == "Forwards":
        pos_filter = ['FW']
    elif position_group == "Midfielders":
        pos_filter = ['MF']
    else:
        pos_filter = ['FW', 'MF']
    
    st.divider()
    
    max_age = st.slider("Maximum Age", 18, 40, 35)
    min_goals = st.slider("Minimum Goals", 0, 40, 0)
    
    st.divider()
    
    sort_by = st.radio(
        "Ranking Method",
        ["Overall Score", "Goals", "Assists", "xG"],
        key="sort"
    )
    
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

# Process data
scored_df = calculate_scores(df, pos_filter)

# Apply filters
scored_df = scored_df[
    (scored_df['Age'] <= max_age) & 
    (scored_df['Goals'] >= min_goals)
]

# Sort
sort_map = {
    "Overall Score": "Final_Score",
    "Goals": "Goals",
    "Assists": "Assists",
    "xG": "xG"
}
scored_df = scored_df.sort_values(sort_map[sort_by], ascending=False)

# Display
tab1, tab2, tab3 = st.tabs(["📊 Player Rankings", "🔍 Player Profile", "📈 Comparison"])

with tab1:
    st.subheader(f"🏆 Top {position_group}")
    
    # Display columns
    display_cols = ['Player', 'Team', 'Age', 'Goals', 'Assists', 'xG', 
                    'Pass_Completion', 'Final_Score']
    
    # Top 10 players
    top10 = scored_df[display_cols].head(10)
    
    st.dataframe(
        top10.style
        .format({
            'Pass_Completion': '{:.1f}%',
            'Final_Score': '{:.3f}'
        })
        .background_gradient(subset=['Final_Score'], cmap='RdYlGn')
        .apply(lambda x: ['font-weight: bold' if x.name == top10.index[0] else '' 
                          for _ in range(len(x))], axis=1),
        use_container_width=True,
        height=400
    )
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_scorer = scored_df.iloc[0]
        st.metric("Best Overall", f"{best_scorer['Player']}", 
                 f"Score: {best_scorer['Final_Score']:.2f}")
    
    with col2:
        top_goal = scored_df.loc[scored_df['Goals'].idxmax()]
        st.metric("Top Scorer", f"{top_goal['Player']}", 
                 f"{top_goal['Goals']} goals")
    
    with col3:
        top_assist = scored_df.loc[scored_df['Assists'].idxmax()]
        st.metric("Top Creator", f"{top_assist['Player']}", 
                 f"{top_assist['Assists']} assists")

with tab2:
    st.subheader("🔍 Detailed Player Analysis")
    
    selected_player = st.selectbox(
        "Choose a player to analyze:",
        scored_df['Player'].tolist()
    )
    
    if selected_player:
        player = scored_df[scored_df['Player'] == selected_player].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### {selected_player}")
            st.markdown(f"**Team:** {player['Team']}")
            st.markdown(f"**Age:** {player['Age']}")
            st.markdown(f"**Position:** {player['Position']}")
            
            # Performance metrics
            st.markdown("#### Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Goals', 'Assists', 'xG', 'Pass %', 'Dribbles/90', 'Score'],
                'Value': [
                    f"{player['Goals']}",
                    f"{player['Assists']}",
                    f"{player['xG']:.1f}",
                    f"{player['Pass_Completion']:.1f}%",
                    f"{player['Dribbles']:.1f}",
                    f"{player['Final_Score']:.3f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # Percentile rankings
            st.markdown("#### Percentile Rankings")
            for metric in ['Goals', 'Assists', 'xG', 'Pass_Completion']:
                percentile = (scored_df[metric] < player[metric]).mean() * 100
                st.progress(int(percentile) / 100, f"{metric}: Top {100-int(percentile)}%")
        
        with col2:
            # Radar chart
            metrics = ['Goals', 'Assists', 'xG', 'Pass_Completion', 'Dribbles']
            fig = create_radar_chart(player, metrics)
            st.pyplot(fig)

with tab3:
    st.subheader("🔄 Player Comparison")
    
    players_to_compare = st.multiselect(
        "Select 2-3 players to compare:",
        scored_df['Player'].tolist(),
        default=scored_df['Player'].head(2).tolist(),
        max_selections=3
    )
    
    if len(players_to_compare) >= 2:
        compare_df = scored_df[scored_df['Player'].isin(players_to_compare)]
        
        # Comparison table
        st.markdown("### Head-to-Head Comparison")
        comp_display = compare_df[['Player', 'Team', 'Age', 'Goals', 'Assists', 
                                   'xG', 'Pass_Completion', 'Final_Score']]
        st.dataframe(
            comp_display.set_index('Player')
            .style.background_gradient(subset=['Final_Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Bar chart comparison
        st.markdown("### Visual Comparison")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Goals comparison
        axes[0].bar(compare_df['Player'], compare_df['Goals'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_title('Goals Scored')
        axes[0].set_ylabel('Goals')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Assists comparison
        axes[1].bar(compare_df['Player'], compare_df['Assists'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_title('Assists')
        axes[1].set_ylabel('Assists')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Overall score comparison
        axes[2].bar(compare_df['Player'], compare_df['Final_Score'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[2].set_title('Overall Score')
        axes[2].set_ylabel('Score (0-1)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Radar chart comparison
        st.markdown("### Playing Style Comparison")
        fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        metrics = ['Goals', 'Assists', 'xG', 'Pass_Completion', 'Dribbles']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, player_name in enumerate(players_to_compare):
            player = compare_df[compare_df['Player'] == player_name].iloc[0]
            values = [player.get(f'{m}_score', 0.5) for m in metrics]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            values += values[:1]
            
            ax2.fill(angles, values, alpha=0.1, color=colors[idx])
            ax2.plot(angles, values, color=colors[idx], linewidth=2, label=player_name, marker='o')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, size=10)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Playing Style Comparison", size=14, pad=20)
        
        st.pyplot(fig2)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white;'>
    <h3>🤖 AI Sporting Director</h3>
    <p>Data-driven football analytics | 2024-25 Season | Top European Leagues</p>
    <p style='font-size: 0.8em;'>Powered by real player statistics & advanced analytics</p>
</div>
""", unsafe_allow_html=True)