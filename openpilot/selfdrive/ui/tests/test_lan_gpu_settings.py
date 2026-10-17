"""Exercise LAN settings without loading raylib, Params, or a real network."""
import ast
import inspect
import ipaddress
from pathlib import Path
from types import SimpleNamespace

import pytest


UI_ROOT = Path(__file__).resolve().parents[1]


def constructor_signature(relative, class_name):
  tree = ast.parse((UI_ROOT / relative).read_text())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
  init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
  required = len(init.args.args) - len(init.args.defaults)
  parameters = [inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  default=inspect.Parameter.empty if i < required else None)
                for i, arg in enumerate(init.args.args) if i]
  parameters.extend(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                     default=inspect.Parameter.empty if default is None else None)
                    for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True))
  return inspect.Signature(parameters)


def test_mici_constructor_keywords_match_current_widgets():
  signatures = {name: constructor_signature(path, name) for name, path in (
    ('BigButton', 'mici/widgets/button.py'), ('BigParamControl', 'mici/widgets/button.py'),
    ('BigInputDialog', 'mici/widgets/dialog.py'), ('BigDialog', 'mici/widgets/dialog.py'))}
  tree = ast.parse((UI_ROOT / 'mici/layouts/settings/langpu.py').read_text())
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in signatures:
      signatures[node.func.id].bind(*[None for _ in node.args], **{kw.arg: None for kw in node.keywords})


def test_mici_hud_preserves_sp_engagement_fade_and_lan_icons():
  """The upstream MADS engagement fade must survive merging LAN HUD support."""
  class Hud:
    def __init__(self):
      self._chestnut_fade_time = 0

    def _update_state(self):
      pass

  state = SimpleNamespace(engaged=False)
  textures = []
  def texture(*args):
    textures.append(args[0])
    return args[0]
  namespace = {'HudRenderer': Hud, 'ui_state': state, 'gui_app': SimpleNamespace(texture=texture),
               'BlindSpotIndicators': lambda: SimpleNamespace(update=lambda: None),
               'rl': SimpleNamespace(Rectangle=object, get_time=lambda: 12.5)}
  path = UI_ROOT / 'sunnypilot/mici/onroad/hud_renderer.py'
  tree = ast.parse(path.read_text())
  classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
  exec(compile(classes, str(path), 'exec'), namespace)
  hud = namespace['HudRendererSP']()
  assert hud._sp_engaged is False
  assert textures == ['icons_mici/lan_gpu.png', 'icons_mici/lan_gpu_green.png', 'icons_mici/lan_gpu_orange.png']
  state.engaged = True
  hud._update_state()
  assert hud._sp_engaged is True and hud._chestnut_fade_time == 12.5
  state.engaged = False
  hud._update_state()
  assert hud._sp_engaged is False and hud._chestnut_fade_time == 0


@pytest.fixture
def panels():
  values, writes, network_calls, dialogs = {}, [], [], []

  class Params:
    def get(self, key):
      return values.get(key)

    def get_bool(self, key, *_):
      return bool(values.get(key, False))

    def put(self, key, value, block=False):
      values[key] = value
      writes.append((key, value, block))

    put_bool = put

    def remove(self, key):
      values.pop(key, None)
      writes.append((key, None, True))

  state = SimpleNamespace(offroad=True, params=Params())
  state.is_offroad = lambda: state.offroad

  class Button:
    def __init__(self, text='', value='', icon=None, scroll=False):
      self.text, self.value = text, value
      self.enabled_source = True

    def set_enabled(self, enabled):
      self.enabled_source = enabled

    def set_click_callback(self, callback):
      self.callback = callback

    def set_value(self, value):
      self.value = value

    def set_state(self, state):
      self.checked = state

  class ParamControl(Button):
    def __init__(self, text, param, toggle_callback=None):
      super().__init__(text)
      self.param, self._checked = param, state.params.get_bool(param)
      self._toggle_callback = toggle_callback

    def _handle_mouse_release(self, mouse_pos):
      self._checked = not self._checked
      if self._toggle_callback:
        self._toggle_callback(self._checked)
      state.params.put_bool(self.param, self._checked, block=True)

  class Widget:
    def __init__(self):
      self._scroller = SimpleNamespace(items=[])
      self._scroller.add_widgets = self._scroller.items.extend

    def _update_state(self):
      pass

  class InputDialog:
    def __init__(self, hint, default_text='', minimum_length=1, confirm_callback=None, auto_return_to_letters=''):
      self.confirm_callback = confirm_callback

  class InputDialogSP:
    def __init__(self, title, sub_title=None, current_text='', param=None, callback=None, min_text_size=0, password_mode=False):
      assert param is None  # A stale dialog must not automatically write a Param.
      self.callback = callback

    def show(self):
      dialogs.append(self)

  class Thread:
    def __init__(self, target, daemon):
      self.target = target

    def start(self):
      self.target()

  def parse_address(text):
    try:
      return ipaddress.IPv4Interface(text)
    except ValueError:
      return None

  def item(**kwargs):
    assert 'param' not in kwargs  # The guarded callback owns persistence.
    action = Button()
    action.set_enabled(kwargs.get('enabled', True))
    return SimpleNamespace(action_item=action, callback=kwargs.get('callback'))

  namespace = {
    'ui_state': state, 'BigButton': Button, 'GreyBigButton': Button, 'BigParamControl': ParamControl,
    'BigInputDialog': InputDialog, 'BigDialog': lambda *args: args, 'NavScroller': Widget, 'Widget': Widget,
    'gui_app': SimpleNamespace(push_widget=dialogs.append),
    'threading': SimpleNamespace(Thread=Thread),
    'usb_network': SimpleNamespace(parse_address=parse_address,
      configured_address=lambda: ipaddress.IPv4Interface(values.get('UsbNetworkAddress', '10.47.0.1/24')),
      up=lambda: network_calls.append(('up', dict(values))), down=lambda: network_calls.append(('down', dict(values)))),
    'tr': lambda value: value, 'toggle_item_sp': item, 'button_item_sp': item,
    'Scroller': lambda items, **kwargs: SimpleNamespace(items=items),
    'DialogResult': SimpleNamespace(CONFIRM='confirm'), 'InputDialogSP': InputDialogSP, 'alert_dialog': lambda value: value,
  }
  for path in ('mici/layouts/settings/langpu.py', 'sunnypilot/layouts/settings/langpu.py'):
    tree = ast.parse((UI_ROOT / path).read_text())
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
    exec(compile(classes, str(UI_ROOT / path), 'exec'), namespace)
  return SimpleNamespace(namespace=namespace, state=state, values=values, writes=writes, network_calls=network_calls, dialogs=dialogs)


def test_mici_toggle_is_offroad_only_and_persists_before_network(panels):
  panel = panels.namespace['LanGpuLayoutMici']()
  toggle = next(item for item in panel._scroller.items if getattr(item, 'param', None) == 'UsbNetworkEnabled')
  panels.state.offroad = False
  toggle._handle_mouse_release(None)
  assert panels.writes == panels.network_calls == []
  panels.state.offroad = True
  toggle._handle_mouse_release(None)
  assert panels.network_calls == [('up', {'UsbNetworkEnabled': True})]
  assert panels.writes == [('UsbNetworkEnabled', True, True)]


@pytest.mark.parametrize('field', ['_server_btn', '_usb_address_btn'])
def test_mici_stale_dialog_cannot_write_or_reconfigure(panels, field):
  panel = panels.namespace['LanGpuLayoutMici']()
  getattr(panel, field).callback()
  panels.state.offroad = False
  panels.dialogs[-1].confirm_callback('10.48.0.1/24')
  assert panels.writes == panels.network_calls == []


def test_mici_invalid_address_is_not_persisted(panels):
  panel = panels.namespace['LanGpuLayoutMici']()
  panel._usb_address_btn.callback()
  panels.dialogs[-1].confirm_callback('not an address')
  assert panels.writes == panels.network_calls == []
  assert panels.dialogs[-1][0] == 'invalid address'


@pytest.mark.parametrize('field', ['_enabled', '_usb_network', '_usb_gateway'])
def test_tici_stale_toggle_cannot_write_or_reconfigure(panels, field):
  panel = panels.namespace['LanGpuLayout']()
  panels.state.offroad = False
  getattr(panel, field).callback(True)
  assert panels.writes == panels.network_calls == []


def test_tici_network_toggle_persists_before_apply(panels):
  panel = panels.namespace['LanGpuLayout']()
  panel._usb_network.callback(True)
  assert panels.network_calls == [('up', {'UsbNetworkEnabled': True})]
  assert panels.writes == [('UsbNetworkEnabled', True, True)]


@pytest.mark.parametrize('callback', ['_edit_server', '_edit_usb_address'])
def test_tici_stale_dialog_cannot_write_or_reconfigure(panels, callback):
  panel = panels.namespace['LanGpuLayout']()
  getattr(panel, callback)()
  panels.state.offroad = False
  panels.dialogs[-1].callback('confirm', '10.48.0.1/24')
  assert panels.writes == panels.network_calls == []
