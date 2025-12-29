from typing import cast, Any, Literal
import numpy as np
import vedo
import vedo.shapes

Color = tuple[float, float, float] | str | np.ndarray


class TrailLine:

    _trail_length: float

    _indices: list[float]
    _pts: list[list[float]]
    _lines: list[vedo.shapes.Line]

    _linewidth: int
    _color: Color
    _alpha_min: float
    _alpha_max: float

    def __init__(self, trail_length: float, linewidth: int, color: Color,
                 alpha_min: float, alpha_max: float):

        self._trail_length = trail_length
        self._linewidth = linewidth
        self._color = color
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

        self._indices = []
        self._pts = []
        self._lines = []

    def add_point(self, index: float, pt: list[float], plotter: vedo.Plotter):

        self._pts.append(pt)
        self._indices.append(index)

        if len(self._pts) > 1:
            new_line = vedo.shapes.Line(self._pts[-1],
                                        self._pts[-2],
                                        c=cast(Any, self._color),
                                        lw=self._linewidth)
            plotter.add(new_line)

            self._lines.append(new_line)

        self.update_trail(index=index, plotter=plotter)

    def update_trail(self, index: float,
                     plotter: vedo.Plotter) -> Literal['not_empty', 'empty']:
        if len(self._pts) > 1:
            cut_at = index - self._trail_length
            while self._indices[0] < cut_at:
                self._indices.pop(0)
                self._pts.pop(0)

                if len(self._lines) > 0:
                    plotter.remove(self._lines[0])
                    self._lines.pop(0)

        # update alpha
        for i, line in enumerate(self._lines):
            if line is None:
                continue

            alpha = self._alpha_min + (
                (self._alpha_max - self._alpha_min) / len(self._lines)) * i
            line.alpha(alpha)

        return 'empty' if len(self._lines) == 0 else 'not_empty'

    def destroy(self, plotter: vedo.Plotter):
        for line in self._lines:
            if line is not None:
                plotter.remove(line)

        self._indices = []
        self._pts = []
        self._lines = []
