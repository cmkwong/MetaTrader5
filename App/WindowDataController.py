from TkWindow import TkWindow


class WindowDataController(TkWindow):

   def run(self, root):
      self.openTopWindowByFrame(root, [], title="Data Controller", windowSize='400x1000')

   def dataFrame(self, root):
      pass