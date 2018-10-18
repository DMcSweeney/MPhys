# Class used to interact with sliced dataset
# Press J to go down a slice and K to go up
import matplotlib.pyplot as plt


class Interact(object):

    @staticmethod
    def multi_slice_viewer(volume):
        # Main function to display results
        Interact.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', Interact.process_key)
        plt.show()

    @staticmethod
    def process_key(event):
        # Function to process a key press
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            Interact.previous_slice(ax)
        elif event.key == 'k':
            Interact.next_slice(ax)
        fig.canvas.draw()

    @staticmethod
    def previous_slice(ax):
        # Function to move to previous slice
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        pass

    @staticmethod
    def next_slice(ax):
        # Move to next slice
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        pass

    @staticmethod
    def remove_keymap_conflicts(new_keys_set):
        # Matplotlib comes with default key bindings, this removes them
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
