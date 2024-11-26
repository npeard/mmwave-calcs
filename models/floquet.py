
class ParametrizedSpinChain:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
class FloquetEngine(ParametrizedSpinChain):
    def __init__(self, param1, param2):
        super().__init__(param1, param2)