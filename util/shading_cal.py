import numpy as np

def calculate_k(x, y, z):
    # x : 851.35  measure idx : 24
    # y : 490.83  measure idx : 0
    # z : 668.79  measure idx : 10

    a = -0.18719133253333503
    b = -0.096854473057347
    c = 0.18723750662193742
    d = 0.026980263739690872

    t = -(a*x+b*y+c*z+d)/(a+b+c)
    meaure_r_p = x+t
    meaure_g_p = y+t
    meaure_b_p = z+t

    distance = ((x- meaure_r_p) ** 2) + ((y - meaure_g_p) ** 2) +((z- meaure_b_p) ** 2)
    distance = distance**0.5

    return distance

