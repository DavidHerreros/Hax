

import inspect
import ast
from typing import get_origin, get_args, Optional, Iterable, Union, Pattern
import re

from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QCheckBox, QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, \
    QMessageBox, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt

from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget


def _is_bool_annot(annot):
    if annot is bool:
        return True
    origin = get_origin(annot)
    if origin is Optional:
        args = get_args(annot)
        return any(a is bool for a in args)
    return False


def _signature_params(cls):
    """Return list of (name, default, annotation, required:bool) for cls.__init__"""
    sig = inspect.signature(cls.__init__)
    out = []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = object() if p.default is inspect._empty else p.default
        annot = p.annotation if p.annotation is not inspect._empty else None
        out.append((name, default, annot, default is object()))
    return out


def _to_text(value):
    if value is object():
        return ""  # required: leave blank
    if value is None:
        return "None"
    return repr(value) if not isinstance(value, (int, float, bool, str)) else str(value)


def _parse_from_text(text, annot, fallback):
    """Try to convert a text field to a Python object."""
    s = text.strip()
    if s == "":
        # Treat blank as "use default" by returning a sentinel; caller can decide.
        return object()
    if s.lower() == "none":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # If annotation is a number, try those first
    try:
        if annot is int:
            return int(s)
        if annot is float:
            return float(s)
    except Exception:
        pass

    # Fall back to literal_eval to support lists/tuples/dicts, etc.
    try:
        return ast.literal_eval(s)
    except Exception:
        # Keep as raw string
        return s


class ParamTableWidget(QWidget):
    """
    A dock widget that shows a 2-col table of constructor kwargs for a selected class.
    Left col: parameter name (read-only).
    Right col: editor widget for the value (bool/int/float/text).
    """
    def __init__(self, parent=None, exclude: Iterable[Union[str, Pattern]] = ("n_components",)):
        super().__init__(parent)

        from sklearn.decomposition import PCA
        from umap import UMAP

        self._methods = dict([("PCA", PCA),] + ([("UMAP", UMAP)]))  # label -> class
        self._current_cls = None
        self._defaults = {}   # name -> default (or _EMPTY)
        self._editors = {}    # name -> widget
        self._exclude_global = list(exclude)

        # -- UI --
        self.setLayout(QVBoxLayout())

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.layout().addWidget(self.table, 1)

        # --- Buttons row (added Reset to defaults) ---
        btns = QHBoxLayout()
        self.layout().addLayout(btns)
        self.btn_reset = QPushButton("Reset to defaults", self)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)

        self.btn_reset.clicked.connect(self.reset_to_defaults)

        # Initialization
        self.table.setVisible(False)
        self.btn_reset.setVisible(False)
        self._current_cls = None
        self._defaults.clear()
        self._editors.clear()

    # ------------- public API -------------
    def current_kwargs(self, *, only_changed=False, include_defaults=True):
        """
        Return dict of current values. If only_changed=True, include only those
        that differ from the class defaults or are required.
        """
        out = {}
        for name, editor in self._editors.items():
            value = self._value_from_editor(name, editor)
            default = self._defaults.get(name, object())

            # Blank text fields return _EMPTY: treat as "use default"
            if value is object():
                value = default

            if only_changed:
                if default is object():
                    # required param: always include
                    out[name] = value
                else:
                    if value != default:
                        out[name] = value
            else:
                if include_defaults or default is object():
                    out[name] = value
        return out

    def reset_to_defaults(self):
        """Set all editors back to the constructor defaults for the current class."""
        for name, editor in self._editors.items():
            default = self._defaults.get(name, object())
            self._apply_value_to_editor(editor, default)

    # ------------- internals -------------
    def _iter_excludes_for(self, cls: type):
        # global exclusions
        for item in self._exclude_global:
            yield item

    def _is_excluded(self, name: str, cls: type) -> bool:
        for item in self._iter_excludes_for(cls):
            if isinstance(item, re.Pattern):
                if item.search(name):
                    return True
            else:
                # string: exact match
                if str(item) == name:
                    return True
        return False

    def _on_method_changed(self, label):
        method = label.split(" ")[-1]
        cls = self._methods.get(method)
        if cls is None:
            # No known method → hide table
            self.table.setVisible(False)
            self.btn_reset.setVisible(False)
            self._current_cls = None
            self._defaults.clear()
            self._editors.clear()
            return
        self.table.setVisible(True)
        self.btn_reset.setVisible(True)
        self._current_cls = cls
        params = _signature_params(cls)
        params = [p for p in params if not self._is_excluded(p[0], cls)]
        self._defaults = {n: d for (n, d, a, r) in params}
        self._editors.clear()

        self.table.setRowCount(len(params))  # Parameter "n_components" is set always by the viewer
        for row, (name, default, annot, required) in enumerate(params):
            # if name != "n_components":
            # name cell
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, name_item)

            # editor cell
            editor = self._make_editor(name, default, annot, required)
            self.table.setCellWidget(row, 1, editor)
            self._editors[name] = editor

        self.table.resizeColumnsToContents()

    def _make_editor(self, name, default, annot, required):
        # Booleans get a checkbox (or a 3-state combo if default is None)
        if _is_bool_annot(annot) or isinstance(default, bool):
            if default is None:
                combo = QtWidgets.QComboBox(self)
                combo.addItems(["None", "True", "False"])
                combo.setCurrentIndex(0)
                return combo
            cb = QCheckBox(self)
            cb.setChecked(bool(default) if default is not object() else False)
            return cb

        # Ints / floats get spin boxes with wide ranges
        if annot is int or isinstance(default, int):
            sb = QSpinBox(self)
            sb.setRange(-2_000_000_000, 2_000_000_000)
            if default is not object():
                sb.setValue(int(default))
            return sb

        if annot is float or isinstance(default, float):
            dsb = QDoubleSpinBox(self)
            dsb.setDecimals(8)
            dsb.setRange(-1e12, 1e12)
            if default is not object():
                dsb.setValue(float(default))
            return dsb

        # Everything else -> QLineEdit with repr(default)
        le = QLineEdit(self)
        if required and default is object():
            le.setPlaceholderText("<required>")
        txt = _to_text(default)
        if txt:
            le.setText(txt)
        return le

    def _apply_value_to_editor(self, editor, value):
        # Applies a supplied 'value' (including _EMPTY) into an editor widget
        if isinstance(editor, QCheckBox):
            editor.setChecked(False if value in (object(), None) else bool(value))
            return
        if isinstance(editor, QtWidgets.QComboBox):
            # tri-state bool combo: "None" | "True" | "False"
            if value in (object(), None):
                editor.setCurrentText("None")
            elif value is True:
                editor.setCurrentText("True")
            else:
                editor.setCurrentText("False")
            return
        if isinstance(editor, QSpinBox):
            editor.setValue(int(0 if value is object() else value))
            return
        if isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(0.0 if value is object() else value))
            return
        if isinstance(editor, QLineEdit):
            editor.setText(_to_text(value))
            return

    def _value_from_editor(self, name, editor):
        default = self._defaults.get(name, object())
        # bool widgets
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QtWidgets.QComboBox):
            t = editor.currentText()
            return _parse_from_text(t, bool, default)

        # numeric widgets
        if isinstance(editor, QSpinBox):
            return int(editor.value())
        if isinstance(editor, QDoubleSpinBox):
            return float(editor.value())

        # text
        if isinstance(editor, QLineEdit):
            annot = None  # we could carry annotations if desired
            return _parse_from_text(editor.text(), annot, default)

        return default

    # --- buttons ---
    def _print_kwargs(self, only_changed=False):
        kw = self.current_kwargs(only_changed=only_changed)
        print(("Changed kwargs:" if only_changed else "All kwargs:"), kw)

    def _instantiate_clicked(self):
        cls = self._current_cls
        kw = self.current_kwargs(only_changed=False)
        # Validate required fields
        missing = [n for n, d in self._defaults.items() if d is object() and kw.get(n, object()) is object()]
        if missing:
            QMessageBox.warning(self, "Missing required parameters",
                                f"Please fill required parameter(s): {', '.join(missing)}")
            return
        # Replace any remaining _EMPTY with None (or drop them)
        clean = {k: (None if v is object() else v) for k, v in kw.items()}
        try:
            instance = cls(**clean)
            QMessageBox.information(self, "Success", f"Instantiated {cls.__name__}:\n{instance!r}")
            # You could emit a signal here with `instance` if you want to use it
            self._last_instance = instance
        except Exception as e:
            QMessageBox.critical(self, "Error instantiating", f"{type(e).__name__}: {e}")