from foampilot.base.openFOAMFile import OpenFOAMFile

class ControlDictFile(OpenFOAMFile):
    """
    A class representing the controlDict file in OpenFOAM.
    
    This class handles the creation and manipulation of the controlDict file which controls
    the runtime behavior of an OpenFOAM simulation. It inherits from OpenFOAMFile and provides
    specific functionality for controlDict parameters.
    
    Attributes:
        application (str): The OpenFOAM application to run (e.g., "simpleFoam").
        startFrom (str): Start time option ("startTime", "firstTime", "latestTime").
        startTime (float): Initial simulation time.
        stopAt (str): Stop condition ("endTime", "writeNow", "noWriteNow", "nextWrite").
        endTime (float): Final simulation time.
        deltaT (float): Time step size.
        writeControl (str): Write control method ("timeStep", "runTime", "adjustableRunTime").
        writeInterval (int): Interval between writing results.
        purgeWrite (int): Number of time directories to keep.
        writeFormat (str): File format for output ("ascii", "binary").
        writePrecision (int): Precision for output data.
        writeCompression (str): Compression for output files ("on", "off").
        timeFormat (str): Time directory naming format ("general", "fixed", "scientific").
        timePrecision (int): Precision for time directory names.
        runTimeModifiable (bool): Whether the dictionary can be modified during runtime.
        functions (dict): Dictionary of function objects for additional runtime controls.
    """
    
    def __init__(self, application="incompressibleFluid", startFrom="startTime", startTime=0,
                 stopAt="endTime", endTime=5000, deltaT=1, writeControl="timeStep",
                 writeInterval=100, purgeWrite=10, writeFormat="ascii", writePrecision=6,
                 writeCompression="off", timeFormat="general", timePrecision=6,
                 runTimeModifiable=True, functions=None):
        """
        Initialize the ControlDictFile with simulation control parameters.
        
        Args:
            application: OpenFOAM solver application (default: "simpleFoam").
            startFrom: Starting time option (default: "startTime").
            startTime: Initial simulation time (default: 0).
            stopAt: Stopping condition (default: "endTime").
            endTime: Final simulation time (default: 5000).
            deltaT: Time step size (default: 1).
            writeControl: Write control method (default: "timeStep").
            writeInterval: Write interval in time steps (default: 100).
            purgeWrite: Number of time directories to keep (default: 10).
            writeFormat: Output file format (default: "ascii").
            writePrecision: Output precision (default: 6).
            writeCompression: Output compression (default: "off").
            timeFormat: Time directory format (default: "general").
            timePrecision: Time directory precision (default: 6).
            runTimeModifiable: Allow runtime modification (default: True).
            functions: Dictionary of function objects (default: None -> empty dict).
        """
        # Initialize functions as empty dict if None provided
        if functions is None:
            functions = {}

        # Call parent class constructor with all parameters
        super().__init__(
            object_name="controlDict",
            application=application,
            startFrom=startFrom,
            startTime=startTime,
            stopAt=stopAt,
            endTime=endTime,
            deltaT=deltaT,
            writeControl=writeControl,
            writeInterval=writeInterval,
            purgeWrite=purgeWrite,
            writeFormat=writeFormat,
            writePrecision=writePrecision,
            writeCompression=writeCompression,
            timeFormat=timeFormat,
            timePrecision=timePrecision,
            runTimeModifiable=runTimeModifiable,
            functions=functions
        )

    def to_dict(self):
        """
        Convert the controlDict parameters to a dictionary.
        
        Returns:
            dict: A dictionary containing all controlDict parameters with their current values.
                  The dictionary structure matches the OpenFOAM controlDict format.
        """
        return {
            'application': self.application,
            'startFrom': self.startFrom,
            'startTime': self.startTime,
            'stopAt': self.stopAt,
            'endTime': self.endTime,
            'deltaT': self.deltaT,
            'writeControl': self.writeControl,
            'writeInterval': self.writeInterval,
            'purgeWrite': self.purgeWrite,
            'writeFormat': self.writeFormat,
            'writePrecision': self.writePrecision,
            'writeCompression': self.writeCompression,
            'timeFormat': self.timeFormat,
            'timePrecision': self.timePrecision,
            'runTimeModifiable': self.runTimeModifiable,
            'functions': self.functions
        }

    @classmethod
    def from_dict(cls, config):
        """
        Create a ControlDictFile instance from a configuration dictionary.
        
        This class method allows creating a ControlDictFile instance by providing a dictionary
        with configuration parameters. Missing parameters will use default values.
        
        Args:
            config (dict): Dictionary containing controlDict parameters. Possible keys:
                - application: OpenFOAM solver application
                - startFrom: Starting time option
                - startTime: Initial simulation time
                - stopAt: Stopping condition
                - endTime: Final simulation time
                - deltaT: Time step size
                - writeControl: Write control method
                - writeInterval: Write interval
                - purgeWrite: Number of time directories to keep
                - writeFormat: Output file format
                - writePrecision: Output precision
                - writeCompression: Output compression
                - timeFormat: Time directory format
                - timePrecision: Time directory precision
                - runTimeModifiable: Runtime modification flag
                - functions: Dictionary of function objects
                
        Returns:
            ControlDictFile: A new instance initialized with the provided or default values.
        """
        # Get each parameter from config or use default value
        application = config.get('application', "incompressibleFluid")
        startFrom = config.get('startFrom', "startTime")
        startTime = config.get('startTime', 0)
        stopAt = config.get('stopAt', "endTime")
        endTime = config.get('endTime', 5000)
        deltaT = config.get('deltaT', 1)
        writeControl = config.get('writeControl', "timeStep")
        writeInterval = config.get('writeInterval', 100)
        purgeWrite = config.get('purgeWrite', 10)
        writeFormat = config.get('writeFormat', "ascii")
        writePrecision = config.get('writePrecision', 6)
        writeCompression = config.get('writeCompression', "off")
        timeFormat = config.get('timeFormat', "general")
        timePrecision = config.get('timePrecision', 6)
        runTimeModifiable = config.get('runTimeModifiable', True)
        functions = config.get('functions', {})

        # Create and return new instance
        return cls(
            application=application,
            startFrom=startFrom,
            startTime=startTime,
            stopAt=stopAt,
            endTime=endTime,
            deltaT=deltaT,
            writeControl=writeControl,
            writeInterval=writeInterval,
            purgeWrite=purgeWrite,
            writeFormat=writeFormat,
            writePrecision=writePrecision,
            writeCompression=writeCompression,
            timeFormat=timeFormat,
            timePrecision=timePrecision,
            runTimeModifiable=runTimeModifiable,
            functions=functions
        )