

import napari
from napari.layers import Points


class CustomPointsLayer(napari.layers.Points):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._fixed_name = kwargs.get("name")  # Store the initial name

    @property
    def name(self):
        return self._fixed_name

    @name.setter
    def name(self, value):
        # Prevent renaming by ignoring any attempts to change the name
        pass

    def add(self, event):
        # Override add to prevent adding new points
        pass

    def drag(self, event):
        # Override drag to prevent adding new points
        pass

    def remove_selected(self):
        # Allow deletion of points
        super().remove_selected()

    def save(self, path):
        points_layer = Points(data=self.data)
        points_layer.metadata = self.metadata
        for key, value in self.properties.items():
            points_layer.properties[key] = value
        points_layer.save(path)


class CustomImageLayer(napari.layers.Image):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._fixed_name = kwargs.get("name")  # Store the initial name

    @property
    def name(self):
        return self._fixed_name

    @name.setter
    def name(self, value):
        # Prevent renaming by ignoring any attempts to change the name
        pass
