from strenum import StrEnum


class Joint(StrEnum):
    RIGHT = 'derecha'
    LEFT = 'izquierda'
    BASE_SPINE = 'espina_base'


class ImuComponents(StrEnum):
    # Angle components
    A = 'a'
    B = 'b'
    G = 'g'

    # Movement components
    X = 'x'
    Y = 'y'
    Z = 'z'

    TIME = 't'
