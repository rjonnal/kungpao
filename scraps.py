        new_state = SensorState(self.n)
        new_state.image = self.image.copy()
        new_state.x_slopes = self.x_slopes.copy()
        new_state.y_slopes = self.y_slopes.copy()
        new_state.x_centroids = self.x_centroids.copy()
        new_state.y_centroids = self.y_centroids.copy()
        new_state.box_maxes = self.box_maxes.copy()
        new_state.box_mins = self.box_mins.copy()
        new_state.box_means = self.box_means.copy()
        new_state.box_backgrounds = self.box_backgrounds.copy()
        new_state.error = self.error
        new_state.tip = self.tip
        new_state.tilt = self.tilt
        return new_state
