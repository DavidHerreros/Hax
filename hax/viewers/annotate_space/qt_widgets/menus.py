

from collections import OrderedDict
from functools import partial

from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtGui import QIntValidator, QIcon
from PyQt5.QtWidgets import QMenu, QApplication, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QVBoxLayout, QLabel, \
    QCheckBox

from qtpy.QtWidgets import QWidget

from hax.viewers.annotate_space.utils.utils import getImagePath


def install_canvas_context_menu(viewer):
    """Install and return a controller you can modify later."""
    qt_canvas = viewer.window._qt_viewer.canvas.native
    qt_canvas.setContextMenuPolicy(Qt.NoContextMenu)  # suppress defaults
    ctrl = CanvasCtxMenuController(viewer)
    qt_canvas.installEventFilter(ctrl)
    qt_canvas._napari_ctx_controller = ctrl  # keep alive
    return ctrl

class CanvasCtxMenuController(QObject):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        # Ordered registry of actions we (re)build into the menu each time
        # Each item: id -> dict(text=..., callback=callable, checkable=bool, checked_getter=callable|None, separator=bool)
        self._items = OrderedDict()

        # --- sensible defaults you can override later with set_action_callback ---
        self.add_action("selection_to_chimerax", "Open selection with ChimeraX", callback=None)

    # ---------- public API ----------
    def add_action(self, id, text, *, callback=None, checkable=False, checked_getter=None):
        """Register a menu entry. If callback is None you can set it later."""
        self._items[id] = dict(
            text=text, callback=callback, checkable=checkable,
            checked_getter=checked_getter, separator=False
        )
        return id

    def add_separator(self, id="sep"):
        self._items[id] = dict(text=None, callback=None, checkable=False,
                               checked_getter=None, separator=True)
        return id

    def set_action_callback(self, id, callback):
        """Replace/attach a callback later on."""
        if id in self._items:
            self._items[id]["callback"] = callback
        else:
            raise KeyError(f"Action id '{id}' not found")

    def remove_action(self, id):
        self._items.pop(id, None)

    # ---------- internals ----------
    def _build_and_exec_menu(self, parent, global_pos):
        menu = QMenu(parent)
        qa_by_id = {}
        for aid, spec in self._items.items():
            if spec["separator"]:
                menu.addSeparator()
                continue
            act = menu.addAction(spec["text"])
            if spec["checkable"]:
                act.setCheckable(True)
                if spec["checked_getter"] is not None:
                    try:
                        act.setChecked(bool(spec["checked_getter"](self.viewer)))
                    except Exception:
                        pass
            # Use partial to bind id (avoid late-binding in loops)
            act.triggered.connect(partial(self._trigger, aid))
            qa_by_id[aid] = act
        chosen = menu.exec_(global_pos)
        return chosen, qa_by_id

    def _trigger(self, action_id, checked=False):
        spec = self._items.get(action_id)
        cb = spec and spec.get("callback")
        if cb is None:
            return
        # Call signature: cb(viewer) or cb(viewer, checked) for checkables
        try:
            if spec.get("checkable"):
                cb(self.viewer, checked)
            else:
                cb(self.viewer)
        except TypeError:
            # Fallback if user provided a cb(viewer) for a checkable action
            cb(self.viewer)

    # ---------- event filter ----------
    def eventFilter(self, obj, event):
        # Native context-menu event (trackpad two-finger, keyboard menu key, etc.)
        if event.type() == QEvent.ContextMenu:
            self._build_and_exec_menu(obj, event.globalPos())
            return True  # consume

        # Swallow right-press so VisPy camera doesn't zoom, but DON'T open here
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            return True

        # Open on right-button release (for classic mouse right-click)
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            # If a popup is already open (because ContextMenu fired), don't open again
            if QApplication.activePopupWidget() is None:
                self._build_and_exec_menu(obj, event.globalPos())
            return True

        return False

class SavingMenuWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.save_btn = QPushButton("Save selections")

        layout_button_save_screenshot = QHBoxLayout()
        self.save_screenshot_button = QPushButton("Save landscape screenshot")
        self.screenshot_dpi = QLineEdit()
        self.screent_dpi_label = QLabel("Screenshot DPI")
        self.transparent_background = QCheckBox(text="Make background transparent?")
        onlyInt = QIntValidator()
        onlyInt.setBottom(1)
        self.screenshot_dpi.setValidator(onlyInt)
        layout_button_save_screenshot.addWidget(self.save_screenshot_button)
        layout_button_save_screenshot.addWidget(self.screent_dpi_label)
        layout_button_save_screenshot.addWidget(self.screenshot_dpi)
        layout_button_save_screenshot.addWidget(self.transparent_background)
        widget_save_screenshot = QWidget()
        widget_save_screenshot.setLayout(layout_button_save_screenshot)

        layout_main = QVBoxLayout()
        layout_main.addWidget(self.save_btn)
        layout_main.addWidget(widget_save_screenshot)
        layout_main.addStretch(1)
        self.setLayout(layout_main)

class ClusteringMenuWidget(QWidget):
    def __init__(self, ndims):
        super().__init__()
        dims = [f"Dim {dim + 1}" for dim in range(ndims)]
        self.cluster_btn = QPushButton("Compute KMeans")
        self.cluster_num = QLineEdit()
        self.dimension_sel = QComboBox()
        _ = [self.dimension_sel.addItem(item) for item in dims]
        self.dimension_sel.setCurrentIndex(0)
        self.dimension_btn = QPushButton("Cluster along PCA dimension")
        self.morph_button = QPushButton("Morph ChimeraX")
        self.morph_button.setIcon(QIcon(getImagePath("chimerax_logo.png")))
        onlyInt = QIntValidator()
        onlyInt.setBottom(1)
        self.cluster_num.setValidator(onlyInt)
        layout_main = QVBoxLayout()
        layout_cluster = QHBoxLayout()
        widget_cluster = QWidget()
        layout_button = QHBoxLayout()
        widget_button = QWidget()
        layout_cluster.addWidget(self.cluster_num)
        layout_cluster.addWidget(self.dimension_sel)
        layout_button.addWidget(self.cluster_btn)
        layout_button.addWidget(self.dimension_btn)
        widget_cluster.setLayout(layout_cluster)
        widget_button.setLayout(layout_button)
        layout_main.addWidget(widget_cluster)
        layout_main.addWidget(widget_button)
        layout_main.addWidget(self.morph_button)
        layout_main.addStretch(1)
        self.setLayout(layout_main)