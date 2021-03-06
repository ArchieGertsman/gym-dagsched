from dataclasses import dataclass


@dataclass
class Worker:
    id_: int
    type_: int
    task = None


    @property
    def available(self):
        return self.task == None


    def make_available(self):
        self.task = None


    def compatible_with(self, op):
        return self.type_ in op.compatible_worker_types


    def can_assign(self, op):
        return self.available and self.compatible_with(op)

