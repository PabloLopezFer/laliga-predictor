#!/usr/bin/env python3
"""Servidor proxy NUEVO - Sin caché"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.request
import urllib.error
import os
import math

# Cargar variables de entorno desde .env (solo en local)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Carga el archivo .env
    print("✅ Variables de entorno cargadas desde .env")
except ImportError:
    # En producción (Render/Railway) no necesita dotenv
    print("ℹ️ python-dotenv no instalado, usando variables del sistema")

# Leer de variables de entorno
# En local: leer de archivo .env
# En producción (Render/Railway): configuradas en la plataforma
API_KEY = os.getenv('API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
PORT = int(os.getenv('PORT', 8001))

# Validar que las keys están configuradas
if not API_KEY or not CLAUDE_API_KEY:
    print("❌ ERROR: Variables de entorno faltantes!")
    print("Por favor, configura API_KEY y CLAUDE_API_KEY")
    print("Ver COMO_DESPLEGAR.md para instrucciones")
    exit(1)

# Mapeo de nombres
NAME_MAP = {
    'Barcelona': 'FC Barcelona',
    'Real Madrid': 'Real Madrid CF',
    'Atlético Madrid': 'Club Atlético de Madrid',
    'Sevilla': 'Sevilla FC',
    'Real Betis': 'Real Betis Balompié',
    'Villarreal': 'Villarreal CF',
    'Girona': 'Girona FC',
    'Espanyol': 'RCD Espanyol de Barcelona',
    'Mallorca': 'RCD Mallorca',
    'Athletic Club': 'Athletic Club',
    'Osasuna': 'CA Osasuna',
    'Real Oviedo': 'Real Oviedo',
    'Rayo Vallecano': 'Rayo Vallecano de Madrid',
    'Valencia': 'Valencia CF',
    'Getafe': 'Getafe CF',
    'Alavés': 'Deportivo Alavés',
    'Real Sociedad': 'Real Sociedad de Fútbol',
    'Celta': 'RC Celta de Vigo',
    'Levante': 'Levante UD',
    'Elche': 'Elche CF',
    'Las Palmas': 'UD Las Palmas',
    'Leganés': 'CD Leganés',
    'Valladolid': 'Real Valladolid CF'
}

# ============================================================================
# FUNCIONES AUXILIARES PARA CÁLCULO DE PROBABILIDADES DE GOLES (POISSON)
# ============================================================================

def poisson_probability(k, lambda_value):
    """
    Calcula la probabilidad de Poisson para exactamente k eventos
    P(X = k) = (λ^k × e^-λ) / k!
    """
    return (lambda_value ** k * math.exp(-lambda_value)) / math.factorial(k)

def calculate_over_probability(lambda_total, threshold):
    """
    Calcula la probabilidad de que haya MÁS de X goles
    P(over X) = 1 - P(0) - P(1) - ... - P(X)
    """
    prob_under = sum(poisson_probability(k, lambda_total) for k in range(int(threshold) + 1))
    return max(0, min(100, (1 - prob_under) * 100))

def calculate_btts_probability(lambda_home, lambda_away):
    """
    Calcula la probabilidad de que ambos equipos marquen
    P(BTTS) = P(home >= 1) × P(away >= 1)
    """
    prob_home_scores = 1 - poisson_probability(0, lambda_home)
    prob_away_scores = 1 - poisson_probability(0, lambda_away)
    return max(0, min(100, prob_home_scores * prob_away_scores * 100))

def calculate_match_probabilities(lambda_home, lambda_away, max_goals=7):
    """
    Calcula probabilidades de victoria local, empate y victoria visitante
    usando distribución de Poisson con ajuste natural de incertidumbre
    
    El fútbol tiene incertidumbre inherente - aplicamos un factor de regresión
    para reflejar que las diferencias extremas de calidad rara vez se traducen
    en certezas del 100% (lesiones, suerte, errores arbitrales, etc.)
    
    Args:
        lambda_home: Goles esperados del equipo local
        lambda_away: Goles esperados del equipo visitante
        max_goals: Máximo de goles a considerar en los cálculos (default 7)
    
    Returns:
        dict con probabilidades de winHome, draw, winAway
    """
    # AJUSTE NATURAL: Reducir diferencias extremas de lambdas
    # En fútbol real, diferencias de calidad se ven reducidas por factores
    # impredecibles (lesiones, suerte, motivación, etc.)
    
    # Calcular la diferencia de goles esperados
    lambda_diff = abs(lambda_home - lambda_away)
    
    # Si la diferencia es muy grande (>3 goles), aplicar regresión a la media
    # Esto refleja que en fútbol, las sorpresas pasan con más frecuencia
    # de lo que predicen modelos puramente matemáticos
    if lambda_diff > 3:
        # Factor de regresión: cuanto mayor la diferencia, más regresión
        regression_factor = 0.7  # 30% de regresión hacia la media
        mean_lambda = (lambda_home + lambda_away) / 2
        
        lambda_home = lambda_home * regression_factor + mean_lambda * (1 - regression_factor)
        lambda_away = lambda_away * regression_factor + mean_lambda * (1 - regression_factor)
    
    # Añadir "ruido de incertidumbre" - el fútbol no es 100% predecible
    # Esto aumenta ligeramente ambos lambdas para reflejar varianza
    uncertainty_factor = 0.15
    lambda_home += uncertainty_factor
    lambda_away += uncertainty_factor
    
    prob_home_win = 0
    prob_draw = 0
    prob_away_win = 0
    
    # Calcular todas las combinaciones de resultados posibles
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            # Probabilidad de este resultado exacto
            prob = poisson_probability(home_goals, lambda_home) * poisson_probability(away_goals, lambda_away)
            
            if home_goals > away_goals:
                prob_home_win += prob
            elif home_goals == away_goals:
                prob_draw += prob
            else:
                prob_away_win += prob
    
    # Normalizar para que sumen 100%
    total = prob_home_win + prob_draw + prob_away_win
    if total > 0:
        prob_home_win = (prob_home_win / total) * 100
        prob_draw = (prob_draw / total) * 100
        prob_away_win = (prob_away_win / total) * 100
    
    # Redondear y asegurar que sumen exactamente 100
    prob_home_win = round(prob_home_win)
    prob_draw = round(prob_draw)
    prob_away_win = 100 - prob_home_win - prob_draw
    
    return {
        'winHome': prob_home_win,
        'draw': prob_draw,
        'winAway': prob_away_win
    }

def get_head_to_head_stats(home_api_name, away_api_name, all_matches, max_seasons_back=2):
    """
    Obtiene estadísticas de enfrentamientos directos (H2H) entre dos equipos
    Limita a las últimas 2 temporadas (aproximadamente últimos 24 meses)
    
    Args:
        home_api_name: Nombre API del equipo local
        away_api_name: Nombre API del equipo visitante
        all_matches: Lista de todos los partidos
        max_seasons_back: Número de temporadas hacia atrás (default 2)
    
    Returns:
        dict con estadísticas de enfrentamientos directos
    """
    from datetime import datetime, timedelta, timezone
    
    # Filtrar partidos entre estos dos equipos
    h2h_matches = [m for m in all_matches if 
                   (m['homeTeam']['name'] == home_api_name and m['awayTeam']['name'] == away_api_name) or
                   (m['homeTeam']['name'] == away_api_name and m['awayTeam']['name'] == home_api_name)]
    
    # Limitar a las últimas 2 temporadas (aproximadamente 730 días)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=365 * max_seasons_back)
    h2h_recent = [m for m in h2h_matches if datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00')) > cutoff_date]
    
    if not h2h_recent:
        return {
            'matches': 0,
            'homeAvgScored': 0,
            'homeAvgConceded': 0,
            'awayAvgScored': 0,
            'awayAvgConceded': 0,
            'totalGoalsAvg': 0,
            'h2hExists': False
        }
    
    # Calcular estadísticas
    home_goals = []
    away_goals = []
    
    for m in h2h_recent:
        if m['homeTeam']['name'] == home_api_name:
            home_goals.append(m['score']['fullTime']['home'])
            away_goals.append(m['score']['fullTime']['away'])
        else:
            home_goals.append(m['score']['fullTime']['away'])
            away_goals.append(m['score']['fullTime']['home'])
    
    return {
        'matches': len(h2h_recent),
        'homeAvgScored': sum(home_goals) / len(home_goals) if home_goals else 0,
        'homeAvgConceded': sum(away_goals) / len(away_goals) if away_goals else 0,
        'awayAvgScored': sum(away_goals) / len(away_goals) if away_goals else 0,
        'awayAvgConceded': sum(home_goals) / len(home_goals) if home_goals else 0,
        'totalGoalsAvg': (sum(home_goals) + sum(away_goals)) / len(h2h_recent) if h2h_recent else 0,
        'h2hExists': True
    }

def calculate_opponent_quality_factor(team_position, team_points, opponent_position, opponent_points):
    """
    Calcula un factor de ajuste basado en la calidad del rival
    
    Args:
        team_position: Posición del equipo en la tabla
        team_points: Puntos del equipo
        opponent_position: Posición del rival
        opponent_points: Puntos del rival
    
    Returns:
        float: Factor multiplicador (0.7 a 1.3)
        - > 1.0 si el rival es más débil (facilita marcar)
        - < 1.0 si el rival es más fuerte (dificulta marcar)
    """
    # Diferencia de posiciones (positiva si el rival es peor)
    position_diff = opponent_position - team_position
    
    # Diferencia de puntos (positiva si tenemos más puntos)
    points_diff = team_points - opponent_points
    
    # Factor basado en posiciones (peso 60%) - MÁS CONSERVADOR
    if position_diff > 10:  # Rival mucho peor
        position_factor = 1.12
    elif position_diff > 5:  # Rival peor
        position_factor = 1.08
    elif position_diff > 0:  # Rival ligeramente peor
        position_factor = 1.03
    elif position_diff > -5:  # Equipos similares
        position_factor = 1.0
    elif position_diff > -10:  # Rival ligeramente mejor
        position_factor = 0.95
    else:  # Rival mucho mejor
        position_factor = 0.88
    
    # Factor basado en puntos (peso 40%) - MÁS CONSERVADOR
    if points_diff > 15:
        points_factor = 1.10
    elif points_diff > 8:
        points_factor = 1.05
    elif points_diff > 0:
        points_factor = 1.02
    elif points_diff > -8:
        points_factor = 1.0
    elif points_diff > -15:
        points_factor = 0.96
    else:
        points_factor = 0.90
    
    # Combinación 60% posición + 40% puntos
    final_factor = position_factor * 0.6 + points_factor * 0.4
    
    # Limitar entre 0.85 y 1.15 (más conservador que antes)
    return max(0.85, min(1.15, final_factor))

def get_goal_statistics(api_name, all_matches, use_weighted=True):
    """
    Calcula estadísticas detalladas de goles para un equipo con ponderación temporal
    Retorna promedios de goles anotados/recibidos como local y visitante
    
    Ponderación:
    - Últimos 3 partidos: 40% peso
    - Partidos 4-8: 30% peso
    - Resto de temporada: 30% peso
    
    Args:
        api_name: Nombre del equipo en la API
        all_matches: Lista de todos los partidos
        use_weighted: Si True, aplica ponderación temporal
    """
    home_matches = [m for m in all_matches if m['homeTeam']['name'] == api_name]
    away_matches = [m for m in all_matches if m['awayTeam']['name'] == api_name]
    
    def calculate_weighted_average(matches, get_scored, get_conceded):
        """Calcula promedio ponderado de goles"""
        if not matches:
            return 0, 0, 0, 0  # scored, conceded, wins, total_matches
        
        if not use_weighted or len(matches) <= 3:
            # Si hay 3 o menos partidos, usar todos por igual
            scored = sum(get_scored(m) for m in matches)
            conceded = sum(get_conceded(m) for m in matches)
            wins = sum(1 for m in matches if get_scored(m) > get_conceded(m))
            return scored / len(matches), conceded / len(matches), wins, len(matches)
        
        # Dividir en grupos con ponderación
        last_3 = matches[:3]
        next_5 = matches[3:8] if len(matches) > 3 else []
        rest = matches[8:] if len(matches) > 8 else []
        
        # Calcular promedios de cada grupo
        avg_scored_last3 = sum(get_scored(m) for m in last_3) / len(last_3) if last_3 else 0
        avg_conceded_last3 = sum(get_conceded(m) for m in last_3) / len(last_3) if last_3 else 0
        wins_last3 = sum(1 for m in last_3 if get_scored(m) > get_conceded(m))
        
        avg_scored_next5 = sum(get_scored(m) for m in next_5) / len(next_5) if next_5 else 0
        avg_conceded_next5 = sum(get_conceded(m) for m in next_5) / len(next_5) if next_5 else 0
        wins_next5 = sum(1 for m in next_5 if get_scored(m) > get_conceded(m))
        
        avg_scored_rest = sum(get_scored(m) for m in rest) / len(rest) if rest else 0
        avg_conceded_rest = sum(get_conceded(m) for m in rest) / len(rest) if rest else 0
        wins_rest = sum(1 for m in rest if get_scored(m) > get_conceded(m))
        
        # Aplicar ponderación: 40% últimos 3, 30% siguientes 5, 30% resto
        weight_last3 = 0.40
        weight_next5 = 0.30 if next_5 else 0
        weight_rest = 0.30 if rest else 0
        
        # Ajustar pesos si faltan grupos
        if not next_5 and not rest:
            weight_last3 = 1.0
        elif not rest:
            weight_last3 = 0.55
            weight_next5 = 0.45
        
        weighted_scored = (avg_scored_last3 * weight_last3 + 
                          avg_scored_next5 * weight_next5 + 
                          avg_scored_rest * weight_rest)
        
        weighted_conceded = (avg_conceded_last3 * weight_last3 + 
                            avg_conceded_next5 * weight_next5 + 
                            avg_conceded_rest * weight_rest)
        
        total_wins = wins_last3 + wins_next5 + wins_rest
        total_matches = len(matches)
        
        return weighted_scored, weighted_conceded, total_wins, total_matches
    
    # Estadísticas como local
    home_scored, home_conceded, home_wins, home_total = calculate_weighted_average(
        home_matches,
        lambda m: m['score']['fullTime']['home'],
        lambda m: m['score']['fullTime']['away']
    )
    
    # Estadísticas como visitante
    away_scored, away_conceded, away_wins, away_total = calculate_weighted_average(
        away_matches,
        lambda m: m['score']['fullTime']['away'],
        lambda m: m['score']['fullTime']['home']
    )
    
    # Calcular porcentajes de over y BTTS
    def calc_over_btts_stats(matches, is_home):
        total = len(matches)
        if total == 0:
            return {'over05': 0, 'over15': 0, 'over25': 0, 'over35': 0, 'btts': 0}
        
        over05 = over15 = over25 = over35 = btts = 0
        for m in matches:
            home_goals = m['score']['fullTime']['home']
            away_goals = m['score']['fullTime']['away']
            total_goals = home_goals + away_goals
            
            if total_goals > 0.5: over05 += 1
            if total_goals > 1.5: over15 += 1
            if total_goals > 2.5: over25 += 1
            if total_goals > 3.5: over35 += 1
            if home_goals > 0 and away_goals > 0: btts += 1
        
        return {
            'over05': round(over05 / total * 100),
            'over15': round(over15 / total * 100),
            'over25': round(over25 / total * 100),
            'over35': round(over35 / total * 100),
            'btts': round(btts / total * 100)
        }
    
    return {
        'asHome': {
            'matches': home_total,
            'avgScored': round(home_scored, 2),
            'avgConceded': round(home_conceded, 2),
            'totalScored': int(sum(m['score']['fullTime']['home'] for m in home_matches)),
            'totalConceded': int(sum(m['score']['fullTime']['away'] for m in home_matches)),
            'wins': home_wins,
            'winRate': round(home_wins / home_total * 100, 1) if home_total > 0 else 0,
            'percentages': calc_over_btts_stats(home_matches, True)
        },
        'asAway': {
            'matches': away_total,
            'avgScored': round(away_scored, 2),
            'avgConceded': round(away_conceded, 2),
            'totalScored': int(sum(m['score']['fullTime']['away'] for m in away_matches)),
            'totalConceded': int(sum(m['score']['fullTime']['home'] for m in away_matches)),
            'wins': away_wins,
            'winRate': round(away_wins / away_total * 100, 1) if away_total > 0 else 0,
            'percentages': calc_over_btts_stats(away_matches, False)
        }
    }

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/next-matchday':
            """Obtiene la siguiente jornada de LaLiga (se actualiza automáticamente cuando una jornada termina)"""
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                # Obtener todos los partidos
                req = urllib.request.Request(
                    'https://api.football-data.org/v4/competitions/PD/matches',
                    headers={'X-Auth-Token': API_KEY}
                )
                
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read())
                    all_matches = data['matches']
                    
                    # Agrupar por jornada
                    matchdays = {}
                    for match in all_matches:
                        matchday = match['matchday']
                        if matchday not in matchdays:
                            matchdays[matchday] = []
                        matchdays[matchday].append(match)
                    
                    # Encontrar la siguiente jornada a mostrar
                    # Buscar la última jornada donde TODOS los partidos estén FINISHED
                    # Luego mostrar la siguiente jornada
                    last_finished_matchday = None
                    
                    for matchday_num in sorted(matchdays.keys(), reverse=True):
                        matches_in_day = matchdays[matchday_num]
                        # Si TODOS los partidos están FINISHED, esta es la última jornada completada
                        if all(m['status'] == 'FINISHED' for m in matches_in_day):
                            last_finished_matchday = matchday_num
                            break
                    
                    # La próxima jornada es la siguiente a la última completada
                    if last_finished_matchday is not None:
                        next_matchday = last_finished_matchday + 1
                    else:
                        # Si ninguna jornada está completamente finished, usar la primera
                        next_matchday = min(matchdays.keys())
                    
                    if next_matchday not in matchdays:
                        # No hay más jornadas disponibles
                        self.wfile.write(json.dumps({
                            'success': False,
                            'error': 'No hay más jornadas disponibles'
                        }).encode('utf-8'))
                        return
                    
                    # Construir datos de la jornada
                    matches_data = []
                    for match in matchdays[next_matchday]:
                        home_full = match['homeTeam']['name']
                        away_full = match['awayTeam']['name']
                        
                        home_short = next((k for k, v in NAME_MAP.items() if v == home_full), home_full)
                        away_short = next((k for k, v in NAME_MAP.items() if v == away_full), away_full)
                        
                        # Formatear fecha
                        from datetime import datetime
                        date_str = match['utcDate']
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        
                        # Convertir a hora de Madrid (UTC+1)
                        import datetime as dt
                        madrid_time = date_obj + dt.timedelta(hours=1)
                        
                        days_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                        months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                        
                        day_name = days_es[madrid_time.weekday()]
                        month_name = months_es[madrid_time.month - 1]
                        formatted_date = f"{day_name} {madrid_time.day} {month_name}, {madrid_time.hour:02d}:{madrid_time.minute:02d}"
                        
                        matches_data.append({
                            'id': match['id'],
                            'home': home_short,
                            'away': away_short,
                            'date': formatted_date,
                            'status': match['status']
                        })
                    
                    # Determinar fechas de la jornada
                    first_match_date = matchdays[next_matchday][0]['utcDate']
                    last_match_date = matchdays[next_matchday][-1]['utcDate']
                    
                    from datetime import datetime
                    first_date = datetime.fromisoformat(first_match_date.replace('Z', '+00:00'))
                    last_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))
                    
                    months_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                    
                    date_range = f"{first_date.day}-{last_date.day} {months_es[last_date.month - 1]} {last_date.year}"
                    
                    result = {
                        'success': True,
                        'matchday': next_matchday,
                        'dateRange': date_range,
                        'matches': matches_data
                    }
                    
                    self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
                    print(f"✅ Jornada {next_matchday} - {len(matches_data)} partidos")
                    
            except Exception as e:
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
                print(f"❌ Error obteniendo jornada: {e}")
                import traceback
                traceback.print_exc()
        
        elif self.path == '/api/standings':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                req = urllib.request.Request(
                    'https://api.football-data.org/v4/competitions/PD/standings',
                    headers={'X-Auth-Token': API_KEY}
                )
                
                with urllib.request.urlopen(req) as response:
                    api_data = json.loads(response.read())
                    standings = {}
                    
                    for team in api_data['standings'][0]['table']:
                        full_name = team['team']['name']
                        short_name = next((k for k, v in NAME_MAP.items() if v == full_name), full_name)
                        
                        standings[short_name] = {
                            'position': team['position'],
                            'points': team['points'],
                            'playedGames': team['playedGames'],
                            'won': team['won'],
                            'draw': team['draw'],
                            'lost': team['lost']
                        }
                    
                    self.wfile.write(json.dumps({
                        'success': True,
                        'standings': standings
                    }, ensure_ascii=False).encode('utf-8'))
                    print(f"✅ Clasificación: {len(standings)} equipos")
                    
            except Exception as e:
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
                print(f"❌ Error: {e}")
        
        elif self.path.startswith('/api/goal-prediction/'):
            """Calcula predicciones de goles usando Poisson y estadísticas reales"""
            # Formato: /api/goal-prediction/Barcelona/Real%20Madrid
            parts = self.path.split('/')
            if len(parts) >= 5:
                home_team = urllib.parse.unquote(parts[3])
                away_team = urllib.parse.unquote(parts[4])
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                try:
                    home_api_name = NAME_MAP.get(home_team, home_team)
                    away_api_name = NAME_MAP.get(away_team, away_team)
                    
                    # Obtener todos los partidos finalizados
                    req = urllib.request.Request(
                        'https://api.football-data.org/v4/competitions/PD/matches?status=FINISHED',
                        headers={'X-Auth-Token': API_KEY}
                    )
                    
                    with urllib.request.urlopen(req) as response:
                        all_matches = json.loads(response.read())['matches']
                        
                        # Obtener clasificación para calcular factor de calidad del rival
                        req_standings = urllib.request.Request(
                            'https://api.football-data.org/v4/competitions/PD/standings',
                            headers={'X-Auth-Token': API_KEY}
                        )
                        
                        standings_data = {}
                        try:
                            with urllib.request.urlopen(req_standings) as standings_response:
                                standings_json = json.loads(standings_response.read())
                                for team in standings_json['standings'][0]['table']:
                                    standings_data[team['team']['name']] = {
                                        'position': team['position'],
                                        'points': team['points']
                                    }
                        except:
                            pass  # Si falla, continuar sin factor de calidad
                        
                        # Obtener datos de clasificación
                        home_pos = standings_data.get(home_api_name, {}).get('position', 10)
                        home_pts = standings_data.get(home_api_name, {}).get('points', 30)
                        away_pos = standings_data.get(away_api_name, {}).get('position', 10)
                        away_pts = standings_data.get(away_api_name, {}).get('points', 30)
                        
                        # Calcular estadísticas de goles para ambos equipos con ponderación temporal
                        # Usa toda la temporada: 40% últimos 3 + 30% siguientes 5 + 30% resto
                        home_stats_specific = get_goal_statistics(home_api_name, all_matches, use_weighted=True)
                        away_stats_specific = get_goal_statistics(away_api_name, all_matches, use_weighted=True)
                        
                        # GENERALES (forma reciente global) - últimos 5 partidos en total
                        home_matches_all = [m for m in all_matches if m['homeTeam']['name'] == home_api_name or m['awayTeam']['name'] == home_api_name][:5]
                        away_matches_all = [m for m in all_matches if m['homeTeam']['name'] == away_api_name or m['awayTeam']['name'] == away_api_name][:5]
                        
                        # Calcular promedio de goles de forma general
                        home_gf_general = sum(m['score']['fullTime']['home'] if m['homeTeam']['name'] == home_api_name 
                                            else m['score']['fullTime']['away'] for m in home_matches_all) if home_matches_all else 0
                        home_gc_general = sum(m['score']['fullTime']['away'] if m['homeTeam']['name'] == home_api_name 
                                            else m['score']['fullTime']['home'] for m in home_matches_all) if home_matches_all else 0
                        
                        away_gf_general = sum(m['score']['fullTime']['home'] if m['homeTeam']['name'] == away_api_name 
                                            else m['score']['fullTime']['away'] for m in away_matches_all) if away_matches_all else 0
                        away_gc_general = sum(m['score']['fullTime']['away'] if m['homeTeam']['name'] == away_api_name 
                                            else m['score']['fullTime']['home'] for m in away_matches_all) if away_matches_all else 0
                        
                        home_avg_scored_general = home_gf_general / len(home_matches_all) if home_matches_all else 0
                        home_avg_conceded_general = home_gc_general / len(home_matches_all) if home_matches_all else 0
                        away_avg_scored_general = away_gf_general / len(away_matches_all) if away_matches_all else 0
                        away_avg_conceded_general = away_gc_general / len(away_matches_all) if away_matches_all else 0
                        
                        # Promedio de la liga
                        total_goals = sum(m['score']['fullTime']['home'] + m['score']['fullTime']['away'] for m in all_matches)
                        league_avg = total_goals / (2 * len(all_matches)) if all_matches else 1.4
                        
                        # Calcular lambdas con ENFOQUE HÍBRIDO
                        # 65% peso a estadísticas local/visitante específicas + 35% a forma general reciente
                        
                        # Lambda basado en local/visitante
                        lambda_home_specific = (home_stats_specific['asHome']['avgScored'] * away_stats_specific['asAway']['avgConceded']) / league_avg
                        lambda_away_specific = (away_stats_specific['asAway']['avgScored'] * home_stats_specific['asHome']['avgConceded']) / league_avg
                        
                        # Lambda basado en forma general
                        lambda_home_general = (home_avg_scored_general * away_avg_conceded_general) / league_avg
                        lambda_away_general = (away_avg_scored_general * home_avg_conceded_general) / league_avg
                        
                        # COMBINACIÓN PONDERADA BASE: 65% específico + 35% general
                        lambda_home_base = lambda_home_specific * 0.65 + lambda_home_general * 0.35
                        lambda_away_base = lambda_away_specific * 0.65 + lambda_away_general * 0.35
                        
                        # ===== NUEVO: HISTORIAL DIRECTO (H2H) =====
                        h2h_stats = get_head_to_head_stats(home_api_name, away_api_name, all_matches, max_seasons_back=2)
                        
                        # Si existe historial directo, calcularlo y aplicar un peso del 20%
                        if h2h_stats['h2hExists'] and h2h_stats['matches'] >= 2:
                            lambda_home_h2h = h2h_stats['homeAvgScored']
                            lambda_away_h2h = h2h_stats['awayAvgScored']
                            h2h_weight = 0.20  # 20% peso al H2H
                            
                            # Combinar: 80% modelo base + 20% H2H
                            lambda_home_with_h2h = lambda_home_base * (1 - h2h_weight) + lambda_home_h2h * h2h_weight
                            lambda_away_with_h2h = lambda_away_base * (1 - h2h_weight) + lambda_away_h2h * h2h_weight
                        else:
                            # Sin H2H suficiente, usar solo el modelo base
                            lambda_home_with_h2h = lambda_home_base
                            lambda_away_with_h2h = lambda_away_base
                        
                        # ===== NUEVO: FACTOR DE CALIDAD DEL RIVAL =====
                        # Calcula cómo la calidad del rival afecta la capacidad de marcar
                        home_quality_factor = calculate_opponent_quality_factor(home_pos, home_pts, away_pos, away_pts)
                        away_quality_factor = calculate_opponent_quality_factor(away_pos, away_pts, home_pos, home_pts)
                        
                        # Aplicar factores de calidad (ajusta los goles esperados según la fuerza del rival)
                        lambda_home = lambda_home_with_h2h * home_quality_factor
                        lambda_away = lambda_away_with_h2h * away_quality_factor
                        lambda_total = lambda_home + lambda_away
                        
                        # Información adicional para debugging
                        h2h_info = f"H2H: {h2h_stats['matches']} partidos" if h2h_stats['h2hExists'] else "Sin H2H"
                        quality_info = f"Factor calidad: {home_quality_factor:.2f}/{away_quality_factor:.2f}"
                        
                        # Usar home_stats_specific y away_stats_specific para el resto
                        home_stats = home_stats_specific
                        away_stats = away_stats_specific
                        
                        # Calcular probabilidades con Poisson
                        poisson_over05 = round(calculate_over_probability(lambda_total, 0.5))
                        poisson_over15 = round(calculate_over_probability(lambda_total, 1.5))
                        poisson_over25 = round(calculate_over_probability(lambda_total, 2.5))
                        poisson_over35 = round(calculate_over_probability(lambda_total, 3.5))
                        poisson_btts = round(calculate_btts_probability(lambda_home, lambda_away))
                        
                        # Combinar con estadísticas históricas (70% Poisson + 30% histórico)
                        hist_home = home_stats['asHome']['percentages']
                        hist_away = away_stats['asAway']['percentages']
                        
                        # Promedio de estadísticas históricas
                        hist_over05 = (hist_home['over05'] + hist_away['over05']) / 2
                        hist_over15 = (hist_home['over15'] + hist_away['over15']) / 2
                        hist_over25 = (hist_home['over25'] + hist_away['over25']) / 2
                        hist_over35 = (hist_home['over35'] + hist_away['over35']) / 2
                        hist_btts = (hist_home['btts'] + hist_away['btts']) / 2
                        
                        # Combinar ambos métodos (Poisson + histórico)
                        final_over05 = round(poisson_over05 * 0.7 + hist_over05 * 0.3)
                        final_over15 = round(poisson_over15 * 0.7 + hist_over15 * 0.3)
                        final_over25 = round(poisson_over25 * 0.7 + hist_over25 * 0.3)
                        final_over35 = round(poisson_over35 * 0.7 + hist_over35 * 0.3)
                        final_btts = round(poisson_btts * 0.7 + hist_btts * 0.3)
                        
                        # ===== NUEVO: CALCULAR PROBABILIDADES DE VICTORIA CON POISSON =====
                        match_probs = calculate_match_probabilities(lambda_home, lambda_away)
                        
                        # ===== GENERAR REASONING BASADO EN DATOS REALES (NO IA) =====
                        reasoning = {
                            'homeAdvantages': [],
                            'awayAdvantages': [],
                            'keyFactors': []
                        }
                        
                        # Ventajas del equipo local basadas en datos reales
                        if home_stats['asHome']['winRate'] > 50:
                            reasoning['homeAdvantages'].append(f"Buen rendimiento en casa: {home_stats['asHome']['winRate']}% de victorias como local")
                        if home_stats['asHome']['avgScored'] > away_stats['asAway']['avgConceded']:
                            reasoning['homeAdvantages'].append(f"Ataque efectivo: {home_stats['asHome']['avgScored']} goles/partido en casa vs {away_stats['asAway']['avgConceded']} recibidos por rival fuera")
                        if home_pos < away_pos:
                            reasoning['homeAdvantages'].append(f"Mejor posición en la tabla ({home_pos}º vs {away_pos}º)")
                        reasoning['homeAdvantages'].append("Ventaja de jugar en casa con su afición")
                        
                        # Ventajas del equipo visitante basadas en datos reales
                        if away_stats['asAway']['winRate'] > 40:
                            reasoning['awayAdvantages'].append(f"Rendimiento sólido fuera: {away_stats['asAway']['winRate']}% de victorias como visitante")
                        if away_stats['asAway']['avgScored'] > home_stats['asHome']['avgConceded']:
                            reasoning['awayAdvantages'].append(f"Ataque efectivo fuera: {away_stats['asAway']['avgScored']} goles/partido vs {home_stats['asHome']['avgConceded']} recibidos por rival")
                        if away_pos < home_pos:
                            reasoning['awayAdvantages'].append(f"Mejor posición en clasificación ({away_pos}º vs {home_pos}º)")
                        if away_stats['asAway']['avgConceded'] < league_avg:
                            reasoning['awayAdvantages'].append(f"Defensa sólida fuera: solo {away_stats['asAway']['avgConceded']} goles recibidos/partido")
                        
                        # Factores clave del partido
                        reasoning['keyFactors'].append(f"Diferencia clasificación: {abs(home_pos - away_pos)} posiciones ({home_pts} vs {away_pts} puntos)")
                        reasoning['keyFactors'].append(f"{home_team} promedia {home_stats['asHome']['avgScored']} goles en casa, {away_team} marca {away_stats['asAway']['avgScored']} fuera")
                        reasoning['keyFactors'].append(f"Defensas: {home_team} recibe {home_stats['asHome']['avgConceded']} en casa, {away_team} recibe {away_stats['asAway']['avgConceded']} fuera")
                        if h2h_stats['h2hExists']:
                            reasoning['keyFactors'].append(f"Historial directo: {h2h_stats['matches']} partidos en últimas 2 temporadas considerados")
                        
                        result = {
                            'success': True,
                            'homeTeam': home_team,
                            'awayTeam': away_team,
                            'expectedGoals': {
                                'home': round(lambda_home, 2),
                                'away': round(lambda_away, 2),
                                'total': round(lambda_total, 2)
                            },
                            'matchProbabilities': {
                                'winHome': match_probs['winHome'],
                                'draw': match_probs['draw'],
                                'winAway': match_probs['winAway']
                            },
                            'probabilities': {
                                'over05': final_over05,
                                'over15': final_over15,
                                'over25': final_over25,
                                'over35': final_over35,
                                'btts': final_btts
                            },
                            'poissonProbabilities': {
                                'over05': poisson_over05,
                                'over15': poisson_over15,
                                'over25': poisson_over25,
                                'over35': poisson_over35,
                                'btts': poisson_btts
                            },
                            'historicalStats': {
                                'home': home_stats,
                                'away': away_stats
                            },
                            'method': 'Modelo Profesional con Ponderación Temporal',
                            'dataSource': f'Flujo: 📊 Ponderación temporal (40% últimos 3 + 30% sig.5 + 30% resto) → 🏠 Local (65%) + 📈 Forma (35%) → +🔄 H2H (20% si existe) → ×⚖️ Ajuste calidad rival (0.85-1.15)',
                            'h2hData': h2h_stats,
                            'qualityFactors': {
                                'home': round(home_quality_factor, 2),
                                'away': round(away_quality_factor, 2)
                            }
                        }
                        
                        self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
                        print(f"📊 {home_team} ({lambda_home:.2f}) vs {away_team} ({lambda_away:.2f}) | {h2h_info} | {quality_info} | Over2.5: {final_over25}% | BTTS: {final_btts}%")
                        
                except Exception as e:
                    self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
                    print(f"❌ Error predicción goles: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                self.send_error(400)
        
        elif self.path.startswith('/api/team-stats/'):
            team_name = urllib.parse.unquote(self.path.split('/')[-1])
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                api_name = NAME_MAP.get(team_name, team_name)
                
                req = urllib.request.Request(
                    'https://api.football-data.org/v4/competitions/PD/matches?status=FINISHED',
                    headers={'X-Auth-Token': API_KEY}
                )
                
                with urllib.request.urlopen(req) as response:
                    matches = json.loads(response.read())['matches']
                    
                    # Filtrar partidos (COMPARACIÓN EXACTA)
                    team_matches = [m for m in matches 
                                    if m['homeTeam']['name'] == api_name or 
                                       m['awayTeam']['name'] == api_name]
                    
                    # ORDENAR POR FECHA (más recientes primero)
                    team_matches.sort(key=lambda x: x['utcDate'], reverse=True)
                    
                    last_5 = team_matches[:5]
                    home_matches = [m for m in team_matches if m['homeTeam']['name'] == api_name][:10]
                    away_matches = [m for m in team_matches if m['awayTeam']['name'] == api_name][:10]
                    
                    def calc_stats(matches, api_name):
                        wins = draws = losses = gf = gc = 0
                        for m in matches:
                            is_home = m['homeTeam']['name'] == api_name
                            home_score = m['score']['fullTime']['home']
                            away_score = m['score']['fullTime']['away']
                            
                            my_goals = home_score if is_home else away_score
                            their_goals = away_score if is_home else home_score
                            
                            gf += my_goals
                            gc += their_goals
                            
                            if my_goals > their_goals:
                                wins += 1
                            elif my_goals == their_goals:
                                draws += 1
                            else:
                                losses += 1
                        
                        return {'wins': wins, 'draws': draws, 'losses': losses, 
                                'goalsFor': gf, 'goalsAgainst': gc, 'matches': len(matches)}
                    
                    result = {
                        'success': True,
                        'teamName': team_name,
                        'recentForm': calc_stats(last_5, api_name),
                        'homeForm': calc_stats(home_matches, api_name),
                        'awayForm': calc_stats(away_matches, api_name),
                        'lastMatches': [
                            {
                                'date': m['utcDate'][:10],
                                'home': m['homeTeam']['shortName'],
                                'away': m['awayTeam']['shortName'],
                                'score': f"{m['score']['fullTime']['home']}-{m['score']['fullTime']['away']}"
                            }
                            for m in last_5
                        ]
                    }
                    
                    self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
                    rf = result['recentForm']
                    print(f"✅ {team_name}: {rf['wins']}V/{rf['draws']}E/{rf['losses']}D - {rf['goalsFor']} GF/{rf['goalsAgainst']} GC")
                    
            except Exception as e:
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))
                print(f"❌ Error {team_name}: {e}")
        
        else:
            return SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        if self.path == '/api/analyze':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                
                claude_request = json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "temperature": 0.7,
                    "system": request_data.get('system'),
                    "messages": request_data.get('messages')
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=claude_request,
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': CLAUDE_API_KEY,
                        'anthropic-version': '2023-06-01'
                    }
                )
                
                with urllib.request.urlopen(req) as response:
                    self.wfile.write(response.read())
                    print("✅ Análisis IA generado")
                    
            except urllib.error.HTTPError as e:
                self.wfile.write(json.dumps({'error': f'Claude error: {e.code}'}).encode('utf-8'))
                print(f"❌ Claude error: {e.code}")
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                print(f"❌ Error análisis: {e}")
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
server = HTTPServer(('', PORT), Handler)

print("="*70)
print(f"⚽ LaLiga Predictor - SERVIDOR NUEVO (Puerto {PORT})")
print("="*70)
print("\n✨ ESTE ES UN SERVIDOR NUEVO SIN CACHÉ")
print(f"🌐 Abre: http://localhost:{PORT}\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n✅ Servidor detenido")
