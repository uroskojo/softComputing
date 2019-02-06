import math


def dot(v, w):
    x1, y1 = v
    x2, y2 = w
    return x1 * x2 + y1 * y2


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(t1, t2):
    x1, y1 = t1
    x2, y2 = t2
    return (x2 - x1, y2 - y1)


def unit(v):
    x, y = v
    magnituda = length(v)
    return (x / magnituda, y / magnituda)


def distance(t1, t2):
    return length(vector(t1, t2))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x1, y1 = v
    x2, y2 = w
    return (x1 + x2, y1 + y2)


# Linija sa koordinatama start i end i koordinate tacke point
# Funkcija vraca najkracu udaljenost od tacke do linije
# i koordinate najblize tacke na liniji
# 1  Konvertuje liniju u vektor ('line2vec').
# 2  Pravi vektor koji spaja 'start' linije i tacku('point') ('point_vec').
# 3  Duzina line2vec ('line_len').
# 4  Konvertuje line2vec u jedinicni vektor(unit vector) ('line_unit_vec').
# 5  Skaliranje point_vec sa line_len ('point_vec_scaled').
# 6  Racuna 'dot product'  od line_unit_vec i point_vec_scaled ('t').
# 7  't' mora da bude u rangu od 0 do 1
# 8  Koristim 't' da dobijem najblizu tacku na liniji do kraja vektora  point_vec_scaled
# 9  Racunam distancu od 'nearest' do 'point_vec_scaled'.
# 10 Translate nearest back to the start/end line.


def point2line(point, start, end):

    line2vec = vector(start, end)
    point_vec = vector(start, point)
    line_len = length(line2vec)
    line_unit_vec = unit(line2vec)
    point_vec_scaled = scale(point_vec, 1.0 / line_len)
    t = dot(line_unit_vec, point_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line2vec, t)
    dist = distance(nearest, point_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)