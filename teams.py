"""NFL team display names and primary colors, keyed by nflverse abbreviation."""

TEAMS = {
    'ARI': ('Arizona Cardinals', '#97233F'),
    'ATL': ('Atlanta Falcons', '#A71930'),
    'BAL': ('Baltimore Ravens', '#241773'),
    'BUF': ('Buffalo Bills', '#00338D'),
    'CAR': ('Carolina Panthers', '#0085CA'),
    'CHI': ('Chicago Bears', '#C83803'),
    'CIN': ('Cincinnati Bengals', '#FB4F14'),
    'CLE': ('Cleveland Browns', '#FF3C00'),
    'DAL': ('Dallas Cowboys', '#003594'),
    'DEN': ('Denver Broncos', '#FB4F14'),
    'DET': ('Detroit Lions', '#0076B6'),
    'GB': ('Green Bay Packers', '#203731'),
    'HOU': ('Houston Texans', '#03202F'),
    'IND': ('Indianapolis Colts', '#002C5F'),
    'JAX': ('Jacksonville Jaguars', '#006778'),
    'KC': ('Kansas City Chiefs', '#E31837'),
    'LV': ('Las Vegas Raiders', '#5b5f63'),
    'LAC': ('Los Angeles Chargers', '#0080C6'),
    'LA': ('Los Angeles Rams', '#003594'),
    'LAR': ('Los Angeles Rams', '#003594'),
    'MIA': ('Miami Dolphins', '#008E97'),
    'MIN': ('Minnesota Vikings', '#4F2683'),
    'NE': ('New England Patriots', '#002244'),
    'NO': ('New Orleans Saints', '#D3BC8D'),
    'NYG': ('New York Giants', '#0B2265'),
    'NYJ': ('New York Jets', '#125740'),
    'PHI': ('Philadelphia Eagles', '#004C54'),
    'PIT': ('Pittsburgh Steelers', '#FFB612'),
    'SF': ('San Francisco 49ers', '#AA0000'),
    'SEA': ('Seattle Seahawks', '#69BE28'),
    'TB': ('Tampa Bay Buccaneers', '#D50A0A'),
    'TEN': ('Tennessee Titans', '#4B92DB'),
    'WAS': ('Washington Commanders', '#5A1414'),
    'WSH': ('Washington Commanders', '#5A1414'),
}


def team_info(abbr):
    """(name, color) for a team abbreviation, with a neutral fallback."""
    return TEAMS.get(abbr, (abbr, '#888888'))


def _rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _luminance(rgb):
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255


def _lift(hex_color, min_luminance=0.30):
    """Mix a dark team color toward white until it is readable on the dark background."""
    rgb = _rgb(hex_color)
    t = 0.0
    while _luminance(rgb) < min_luminance and t < 0.9:
        t += 0.05
        base = _rgb(hex_color)
        rgb = tuple(round(c + (255 - c) * t) for c in base)
    return '#%02X%02X%02X' % rgb


NEUTRAL_COLOR = '#C9CFC1'


def matchup_colors(away, home):
    """Readable, distinguishable display colors for two teams on the dark theme."""
    away_c, home_c = _lift(team_info(away)[1]), _lift(team_info(home)[1])
    distance = sum((a - b) ** 2 for a, b in zip(_rgb(away_c), _rgb(home_c))) ** 0.5
    if distance < 60:
        home_c = NEUTRAL_COLOR
    return away_c, home_c
