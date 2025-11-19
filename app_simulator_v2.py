"""
🎯 CARREFOUR VOYAGES SIMULATOR V2
Application interactive pour la présentation compétitive
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from model_predictor import CarrefourPredictor, calculate_cluster_simple

# Configuration de la page
st.set_page_config(
    page_title="Carrefour Voyages Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0051a5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0051a5 0%, #00a3e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    # Chemins pour Streamlit Cloud (fichiers à la racine)
    try:
        df_results = pd.read_csv("carrefour_voyages_analysis_results.csv")
    except:
        # Fallback pour exécution locale
        results_path = r"c:\Users\LIOR\Desktop\python\bdd carrefour\03_RESULTATS_ANALYSE"
        df_results = pd.read_csv(f"{results_path}\\carrefour_voyages_analysis_results.csv")
    
    # Créer cluster_profiles à partir des résultats si le fichier n'existe pas
    try:
        df_clusters = pd.read_csv("cluster_profiles.csv")
    except:
        try:
            results_path = r"c:\Users\LIOR\Desktop\python\bdd carrefour\03_RESULTATS_ANALYSE"
            df_clusters = pd.read_csv(f"{results_path}\\cluster_profiles.csv")
        except:
            # Générer les profils à partir des données
            df_clusters = df_results.groupby('Cluster').agg({
                'CA_Total_Agence': 'mean',
                'Effectifs_Totaux': 'mean',
                'Ratio_Web': 'mean',
                'CA_Par_Effectif': 'mean'
            }).reset_index()
            df_clusters.columns = ['Cluster', 'CA_Moyen', 'Effectifs_Moyen', 'Ratio_Web_Moyen', 'Productivite_Moyenne']
    df_importance = pd.read_csv(f"{results_path}\\feature_importance_best.csv")
    
    return df_results, df_clusters, df_importance

@st.cache_resource
def load_predictor():
    """Load the trained ML model"""
    predictor = CarrefourPredictor()
    try:
        predictor.load_model()
        return predictor
    except Exception as e:
        st.warning(f"Modèle ML non chargé: {e}")
        return None
    
    return df_results, df_clusters, df_importance

try:
    df_results, df_clusters, df_importance = load_data()
    predictor = load_predictor()
    st.success("✅ Données et modèle chargés avec succès!")
except Exception as e:
    st.error(f"❌ Erreur de chargement: {e}")
    st.stop()

# Header
st.markdown('<h1 class="main-header">✈️ CARREFOUR VOYAGES SIMULATOR 🎯</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 Simulateur Interactif de Performance d'Agences")
st.markdown("---")

# Sidebar - Navigation
with st.sidebar:
    st.markdown("## 🎮 MENU")
    page = st.radio(
        "Choisissez votre outil :",
        ["🏗️ Créateur d'Agence", "📊 Analyse de Clusters", "🗺️ Carte Interactive", "🎯 Calculateur ROI", "📈 Importance des Variables"],
        label_visibility="collapsed"
    )

# ===== PAGE 1: CRÉATEUR D'AGENCE =====
if page == "🏗️ Créateur d'Agence":
    st.header("🏗️ CRÉATEUR D'AGENCE VIRTUELLE")
    st.markdown("**Testez différents profils d'agence et découvrez leur cluster !**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Paramètres de l'agence")
        
        ca_total = st.slider(
            "💰 Chiffre d'Affaires Total (€)",
            min_value=float(df_results['CA_Total'].min()),
            max_value=float(df_results['CA_Total'].max()),
            value=float(df_results['CA_Total'].median()),
            step=100000.0,
            format="%.0f"
        )
        
        effectifs = st.slider(
            "👥 Effectifs Totaux",
            min_value=int(df_results['Effectifs_Totaux'].min()),
            max_value=int(df_results['Effectifs_Totaux'].max()),
            value=int(df_results['Effectifs_Totaux'].median()),
            step=1
        )
        
        nb_resa = st.slider(
            "📋 Nombre de Réservations",
            min_value=int(df_results['Nombre_Resa_Total'].min()),
            max_value=int(df_results['Nombre_Resa_Total'].max()),
            value=int(df_results['Nombre_Resa_Total'].median()),
            step=50
        )
        
        ratio_web = st.slider(
            "🌐 Ratio Web (%) - Impact direct sur le CA prédit!",
            min_value=0.0,
            max_value=100.0,
            value=float(df_results['Ratio_Web'].median() * 100),
            step=1.0,
            help="Le ratio web est le 2ème facteur le plus important pour prédire le CA. Changez cette valeur pour voir l'impact!"
        ) / 100
        
    with col2:
        st.subheader("📊 Indicateurs Calculés")
        
        # Calculs de base
        productivite = ca_total / effectifs if effectifs > 0 else 0
        ca_moyen = ca_total / nb_resa if nb_resa > 0 else 0
        resa_par_effectif = nb_resa / effectifs if effectifs > 0 else 0
        
        # Prédiction avec le modèle ML si disponible
        predicted_ca = None
        if predictor is not None:
            try:
                # Préparer les inputs pour le modèle
                input_dict = {
                    'Effectifs_Totaux': effectifs,
                    'Anciennete_moyenne_annees': df_results['Anciennete_moyenne_annees'].median(),
                    'Effectifs_CDI': int(effectifs * 0.7),  # Estimation
                    'Effectifs_CDD': int(effectifs * 0.2),
                    'Effectifs_Aternance': int(effectifs * 0.1),
                    'Effectif_Responsable_Agence': 1,
                    'Effectif_Temps_Plein': int(effectifs * 0.8),
                    'Effectif_Temps_Partiel': int(effectifs * 0.2),
                    'Ratio_CDI': 0.7,
                    'Ratio_Temps_Plein': 0.8,
                    'Ratio_Manager': 1.0 / effectifs if effectifs > 0 else 0,
                    'Ratio_Web': ratio_web,
                    'CA_Par_Effectif': productivite,
                    'Nombre_Resa_Total': nb_resa
                }
                
                predicted_ca = predictor.predict_ca(input_dict)
                ca_diff = ((predicted_ca - ca_total) / ca_total * 100) if ca_total > 0 else 0
                
                # Affichage du CA prédit avec mise en avant
                st.markdown("### 🤖 PRÉDICTION IA")
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;'>
                    <h3 style='color: white; margin: 0;'>CA Prédit</h3>
                    <h1 style='color: white; margin: 0.5rem 0; font-size: 2.5rem;'>{predicted_ca:,.0f} €</h1>
                    <p style='color: white; margin: 0; font-size: 1.2rem;'>{ca_diff:+.1f}% vs saisie manuelle</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Prédiction non disponible: {e}")
        
        # Prédiction simple du cluster basée sur les quantiles
        cluster_pred = calculate_cluster_simple(ca_total, nb_resa, effectifs)
        
        # Mapping cluster to name and color
        cluster_mapping = {
            0: ("� BAS POTENTIEL", "#FFA07A"),
            1: ("🟡 EN DÉVELOPPEMENT", "#FFD700"),
            2: ("🟢 SOLIDE", "#90EE90"),
            3: ("🔵 PERFORMANT", "#00A3E0"),
            4: ("⭐ EXCELLENCE", "#9370DB")
        }
        cluster_name, cluster_color = cluster_mapping.get(cluster_pred, ("INCONNU", "#CCCCCC"))
        
        # Affichage des métriques
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("💰 Productivité", f"{productivite:,.0f} €/employé")
            st.metric("📊 CA Moyen/Résa", f"{ca_moyen:,.0f} €")
        with col_b:
            st.metric("👥 Résa/Employé", f"{resa_par_effectif:.1f}")
            st.metric("🌐 Part Web", f"{ratio_web*100:.1f}%")
        
        # Cluster prédit
        st.markdown("---")
        st.markdown(f"""
        <div style='background-color: {cluster_color}; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h2 style='color: #000; margin: 0;'>CLUSTER PRÉDIT</h2>
            <h1 style='color: #000; margin: 0.5rem 0;'>{cluster_name}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique de positionnement
    st.markdown("---")
    st.subheader("📍 Positionnement vs. Benchmark")
    
    fig = px.scatter(
        df_results,
        x='CA_Total',
        y='CA_Par_Effectif',
        color='Cluster',
        size='Nombre_Resa_Total',
        hover_data=['Agence_nom', 'Region'],
        labels={'CA_Total': 'Chiffre d\'Affaires (€)', 'CA_Par_Effectif': 'Productivité (€/employé)'},
        title="Position de votre agence virtuelle vs. toutes les agences",
        color_continuous_scale='Viridis'
    )
    
    # Ajout du point de l'agence virtuelle
    fig.add_scatter(
        x=[ca_total],
        y=[productivite],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
        name='Votre Agence',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 2: ANALYSE DE CLUSTERS =====
elif page == "📊 Analyse de Clusters":
    st.header("📊 ANALYSE DES CLUSTERS")
    
    # Vue d'ensemble des clusters
    st.subheader("🎯 Vue d'ensemble des 4 segments")
    
    cluster_names = {
        0: "🌟 STARS",
        1: "💎 PREMIUM", 
        2: "📈 EN CROISSANCE",
        3: "🎯 POTENTIEL"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    for idx, (cluster_id, col) in enumerate(zip([0, 1, 2, 3], [col1, col2, col3, col4])):
        cluster_data = df_results[df_results['Cluster'] == cluster_id]
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{cluster_names.get(cluster_id, f'Cluster {cluster_id}')}</h3>
                <h4>{len(cluster_data)} agences</h4>
                <p>CA Moyen: {cluster_data['CA_Total'].mean():,.0f} €</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Graphique de distribution
    st.markdown("---")
    st.subheader("📈 Distribution des Clusters")
    
    # Convertir Cluster en string pour utiliser color_discrete_sequence
    df_results['Cluster_str'] = df_results['Cluster'].astype(str)
    
    fig = px.box(
        df_results,
        x='Cluster_str',
        y='CA_Total',
        color='Cluster_str',
        labels={'CA_Total': 'CA Total (€)', 'Cluster_str': 'Cluster'},
        title="Distribution du CA par Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 agences par cluster
    st.markdown("---")
    selected_cluster = st.selectbox("Sélectionnez un cluster pour voir les top 5 agences", [0, 1, 2, 3])
    
    cluster_df = df_results[df_results['Cluster'] == selected_cluster]
    top_agencies = cluster_df.nlargest(5, 'CA_Total')[['Agence_nom', 'CA_Total', 'Effectifs_Totaux', 'CA_Par_Effectif', 'Nombre_Resa_Total']]
    
    st.dataframe(
        top_agencies.style.format({
            'CA_Total': '{:,.0f} €',
            'CA_Par_Effectif': '{:,.0f} €',
            'Nombre_Resa_Total': '{:,.0f}'
        }),
        use_container_width=True
    )

# ===== PAGE 3: CARTE INTERACTIVE =====
elif page == "🗺️ Carte Interactive":
    st.header("🗺️ CARTE INTERACTIVE DES AGENCES")
    st.markdown("**Visualisez la répartition géographique des agences par cluster**")
    
    # Importer la carte HTML
    import os
    map_path = r"c:\Users\LIOR\Desktop\python\bdd carrefour\05_RAPPORTS_FINAUX\CARTE_INTERACTIVE_AGENCES.html"
    
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            map_html = f.read()
        
        # Afficher la carte dans Streamlit
        st.components.v1.html(map_html, height=700, scrolling=True)
        
        st.markdown("---")
        
        # Statistiques par région
        st.subheader("📊 Statistiques par Région")
        
        region_stats = df_results.groupby('Region').agg({
            'CA_Total': ['mean', 'sum'],
            'Agence_nom': 'count',
            'Effectifs_Totaux': 'mean',
            'CA_Par_Effectif': 'mean'
        }).round(0)
        
        region_stats.columns = ['CA Moyen', 'CA Total', 'Nombre Agences', 'Effectifs Moyens', 'Productivité']
        
        st.dataframe(
            region_stats.style.format({
                'CA Moyen': '{:,.0f} €',
                'CA Total': '{:,.0f} €',
                'Productivité': '{:,.0f} €'
            }),
            use_container_width=True
        )
        
        # Graphique régional
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_results.groupby('Region')['CA_Total'].sum().reset_index(),
                x='Region',
                y='CA_Total',
                title='CA Total par Région',
                labels={'CA_Total': 'CA Total (€)'},
                color_discrete_sequence=['#0051a5']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2 = px.pie(
                df_results,
                names='Region',
                title='Répartition des Agences par Région',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.error("❌ Carte non trouvée. Générez-la d'abord avec `create_interactive_map.py`")
        if st.button("🔄 Générer la carte maintenant"):
            with st.spinner("Génération de la carte en cours..."):
                import subprocess
                subprocess.run(['python', '02_SCRIPTS_PYTHON/create_interactive_map.py'])
                st.success("✅ Carte générée ! Rafraîchissez la page.")

# ===== PAGE 4: CALCULATEUR ROI =====
elif page == "🎯 Calculateur ROI":
    st.header("🎯 CALCULATEUR D'IMPACT ROI")
    st.markdown("**Simulez l'impact des recommandations sur vos KPIs avec le modèle IA**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Agence de Référence")
        
        # Valeurs de référence
        effectif_base = st.number_input(
            "👥 Effectifs Actuels",
            min_value=1,
            max_value=50,
            value=int(df_results['Effectifs_Totaux'].median()),
            step=1
        )
        
        resa_base = st.number_input(
            "📋 Réservations Actuelles",
            min_value=100,
            max_value=10000,
            value=int(df_results['Nombre_Resa_Total'].median()),
            step=100
        )
        
        ratio_web_base = st.slider(
            "🌐 Ratio Web Actuel (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(df_results['Ratio_Web'].median() * 100),
            step=1.0
        )
        
        st.markdown("---")
        st.subheader("⚙️ Modifications à Tester")
        
        delta_effectif = st.slider(
            "👥 Changement Effectifs",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            help="Nombre d'employés à ajouter (+) ou retirer (-)"
        )
        
        delta_resa = st.slider(
            "📋 Changement Réservations (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Augmentation ou diminution du nombre de réservations"
        )
        
        delta_web = st.slider(
            "🌐 Changement Ratio Web (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            help="Augmentation ou diminution de la part du web"
        )
    
    with col2:
        st.subheader("💰 Impact Projeté (IA)")
        
        if predictor is not None:
            try:
                # Calcul scénario de base
                ratio_web_decimal = ratio_web_base / 100
                productivite_base = df_results['CA_Par_Effectif'].median()
                
                input_base = {
                    'Effectifs_Totaux': effectif_base,
                    'Anciennete_moyenne_annees': df_results['Anciennete_moyenne_annees'].median(),
                    'Effectifs_CDI': int(effectif_base * 0.7),
                    'Effectifs_CDD': int(effectif_base * 0.2),
                    'Effectifs_Aternance': int(effectif_base * 0.1),
                    'Effectif_Responsable_Agence': 1,
                    'Effectif_Temps_Plein': int(effectif_base * 0.8),
                    'Effectif_Temps_Partiel': int(effectif_base * 0.2),
                    'Ratio_CDI': 0.7,
                    'Ratio_Temps_Plein': 0.8,
                    'Ratio_Manager': 1.0 / effectif_base if effectif_base > 0 else 0,
                    'Ratio_Web': ratio_web_decimal,
                    'CA_Par_Effectif': productivite_base,
                    'Nombre_Resa_Total': resa_base
                }
                
                ca_base = predictor.predict_ca(input_base)
                
                # Calcul scénario modifié
                nouvel_effectif = effectif_base + delta_effectif
                nouvelles_resa = int(resa_base * (1 + delta_resa / 100))
                nouveau_ratio_web = min(100, max(0, ratio_web_base + delta_web)) / 100
                
                input_nouveau = {
                    'Effectifs_Totaux': nouvel_effectif,
                    'Anciennete_moyenne_annees': df_results['Anciennete_moyenne_annees'].median(),
                    'Effectifs_CDI': int(nouvel_effectif * 0.7),
                    'Effectifs_CDD': int(nouvel_effectif * 0.2),
                    'Effectifs_Aternance': int(nouvel_effectif * 0.1),
                    'Effectif_Responsable_Agence': 1,
                    'Effectif_Temps_Plein': int(nouvel_effectif * 0.8),
                    'Effectif_Temps_Partiel': int(nouvel_effectif * 0.2),
                    'Ratio_CDI': 0.7,
                    'Ratio_Temps_Plein': 0.8,
                    'Ratio_Manager': 1.0 / nouvel_effectif if nouvel_effectif > 0 else 0,
                    'Ratio_Web': nouveau_ratio_web,
                    'CA_Par_Effectif': productivite_base,
                    'Nombre_Resa_Total': nouvelles_resa
                }
                
                ca_nouveau = predictor.predict_ca(input_nouveau)
                
                # Calcul des impacts
                gain_ca = ca_nouveau - ca_base
                pourcent_gain = (gain_ca / ca_base * 100) if ca_base > 0 else 0
                
                productivite_base_calc = ca_base / effectif_base if effectif_base > 0 else 0
                productivite_nouveau = ca_nouveau / nouvel_effectif if nouvel_effectif > 0 else 0
                gain_productivite = productivite_nouveau - productivite_base_calc
                
                # Affichage
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #00a3e0 0%, #0051a5 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;'>
                    <h3 style='color: white; margin: 0;'>CA de Base</h3>
                    <h1 style='color: white; margin: 0.5rem 0; font-size: 2rem;'>{ca_base:,.0f} €</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;'>
                    <h3 style='color: white; margin: 0;'>CA Après Modifications</h3>
                    <h1 style='color: white; margin: 0.5rem 0; font-size: 2rem;'>{ca_nouveau:,.0f} €</h1>
                    <p style='color: white; margin: 0; font-size: 1.2rem;'>{pourcent_gain:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("💰 Gain CA", f"{gain_ca:,.0f} €", f"{pourcent_gain:+.1f}%")
                    st.metric("📊 Productivité Avant", f"{productivite_base_calc:,.0f} €/emp")
                
                with col_b:
                    st.metric("� Effectifs", f"{nouvel_effectif}", f"{delta_effectif:+d}")
                    st.metric("📊 Productivité Après", f"{productivite_nouveau:,.0f} €/emp", 
                             f"{(gain_productivite/productivite_base_calc*100):+.1f}%")
                
                # Analyse
                st.markdown("---")
                st.subheader("🎯 Analyse")
                
                if gain_ca > 0:
                    st.success(f"✅ Cette stratégie génère un gain de **{gain_ca:,.0f} €** ({pourcent_gain:.1f}%)")
                elif gain_ca < 0:
                    st.error(f"❌ Cette stratégie entraîne une perte de **{abs(gain_ca):,.0f} €** ({pourcent_gain:.1f}%)")
                else:
                    st.info("➡️ Aucun impact significatif détecté")
                
                # Recommandations
                if delta_web > 0 and gain_ca > 0:
                    st.info("💡 Le ratio web a un impact positif ! Continuez la digitalisation.")
                if delta_resa > 0 and gain_ca > 0:
                    st.info("💡 L'augmentation des réservations est le levier principal !")
                if delta_effectif > 0 and gain_productivite < 0:
                    st.warning("⚠️ L'ajout d'effectifs réduit la productivité. Optimisez d'abord les process.")
                
            except Exception as e:
                st.error(f"Erreur de calcul: {e}")
        else:
            st.error("❌ Modèle ML non disponible")

# ===== PAGE 4: IMPORTANCE DES VARIABLES =====
elif page == "📈 Importance des Variables":
    st.header("📈 IMPORTANCE DES VARIABLES (RANDOM FOREST)")
    
    st.markdown("**Top 15 des variables les plus importantes pour prédire le CA**")
    
    # Graphique d'importance
    top_features = df_importance.nlargest(15, 'importance_mean')
    
    fig = px.bar(
        top_features,
        x='importance_mean',
        y='feature',
        orientation='h',
        labels={'importance_mean': 'Importance (%)', 'feature': 'Variable'},
        title="Top 15 des Variables par Importance",
        color='importance_mean',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.subheader("📋 Tableau détaillé")
    st.dataframe(
        df_importance.style.format({
            'Importance': '{:.4f}'
        }),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 <b>Carrefour Voyages Simulator</b> | Développé pour la compétition B2</p>
    <p>📊 Données: 73 agences | 38 variables | 4 clusters identifiés</p>
</div>
""", unsafe_allow_html=True)
