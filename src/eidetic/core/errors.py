class EideticError(Exception):
    """Base exception for Eidetic."""


class PluginNotFoundError(EideticError):
    def __init__(self, system: str):
        super().__init__(f"Memory plugin '{system}' was not found.")
        self.system = system


class DependencyMissingError(EideticError):
    def __init__(self, system: str, missing: list[str], install_hint: str):
        missing_str = ", ".join(missing)
        super().__init__(
            f"Missing dependencies for plugin '{system}': {missing_str}. "
            f"Install with: {install_hint}"
        )
        self.system = system
        self.missing = missing
        self.install_hint = install_hint


class CapabilityNotSupportedError(EideticError):
    def __init__(self, system: str, capability: str):
        super().__init__(f"Plugin '{system}' does not support capability '{capability}'.")
        self.system = system
        self.capability = capability


class BackendOperationError(EideticError):
    def __init__(self, system: str, operation: str, message: str):
        super().__init__(f"Plugin '{system}' failed during '{operation}': {message}")
        self.system = system
        self.operation = operation
        self.message = message
