import asyncio
import multiprocessing as mp
from shapely.geometry import box, shape
from typing import List, Dict, Any, Tuple
from shapely import STRtree

class SpatialWorkerProcess:
    """
    Worker process for handling spatial operations with its own dedicated CPU resources.
    Uses multiprocessing queues for communication with the main process.
    """
    def __init__(self, highway_data: Dict[str, Any]):
        self.highway_data = highway_data
        self.cmd_queue: mp.Queue = mp.Queue()
        self.res_queue: mp.Queue = mp.Queue()
        self.process = mp.Process(target=self._run)
        self.process.daemon = True
        self.process.start()
        
    def _run(self) -> None:
        """Main worker process loop that handles incoming commands"""

        # Create the spatial index
        print("Worker process: Creating spatial index...")
        highway_tree, geometries = self._create_highway_strtree(self.highway_data)
        print(f"Worker process: Created STRTree with {len(geometries)} highway geometries")
        
        while True:
            cmd, args = self.cmd_queue.get()
            if cmd == "stop":
                print("Worker process: Shutting down")
                break
            elif cmd == "query":
                bboxes = args
                results = self._process_bboxes(highway_tree, bboxes)
                self.res_queue.put(results)
            else:
                print(f"Worker process: Unknown command '{cmd}'")
                self.res_queue.put(None)
    
    def _create_highway_strtree(self, highway_data: Dict[str, Any]) -> Tuple[Any, List]:
        """Create an STRTree from highway features"""
        # Convert GeoJSON features to Shapely geometries
        geometries = []
        for i, feature in enumerate(highway_data["features"]):
            try:
                if "geometry" in feature:
                    # Create a Shapely geometry from the GeoJSON geometry
                    geom = shape(feature["geometry"])
                    # Store the original feature index with the geometry
                    geometries.append((geom, i))
            except Exception as e:
                print(f"Error processing geometry: {e}")
                continue

        # Extract just the geometries for the tree
        geoms_only = [g for g, _ in geometries]
        
        # Create the STRTree from the geometries
        tree = STRtree(geoms_only)
        
        return tree, geometries
    
    def _process_bboxes(self, highway_tree, bboxes):
        """Process a batch of bounding boxes and return nearby highways for each"""
        results = []
        
        for bbox in bboxes:
            # Create a shapely box from the bounding box coordinates
            query_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Query the STRtree for highways that intersect this box
            nearby_indices = highway_tree.query(query_box)
            
            # Get the actual highway features using the indices from the original highway_data
            nearby_features = []
            for idx in nearby_indices:
                # Get the original feature from highway_data
                feature = self.highway_data["features"][idx]
                nearby_features.append(feature)
            
            results.append(nearby_features)
        
        return results

class AsyncSpatialService:
    """
    Asynchronous service that interfaces with a dedicated spatial worker process.
    Provides async methods for spatial operations that don't block the main event loop.
    """
    def __init__(self, highway_data: Dict[str, Any]):
        self.worker = SpatialWorkerProcess(highway_data)
        print("AsyncSpatialService: Initialized with worker process")
        
    async def find_nearby_highways(self, segment_bboxes) -> List[List[Dict]]:
        """
        Asynchronously find highways near each segment using the worker process.
        
        Args:
            segment_bboxes: JAX array or numpy array of bounding boxes, 
                            each in format (min_lng, min_lat, max_lng, max_lat)
        
        Returns:
            List of lists, where each inner list contains the highway features 
            near the corresponding segment
        """
        # Convert JAX array to numpy if needed, then to a list of tuples
        if hasattr(segment_bboxes, 'tolist'):  # Check if it's a JAX or numpy array
            bboxes_list = segment_bboxes.tolist()
        else:
            bboxes_list = list(segment_bboxes)
        
        # Put the query in the command queue
        self.worker.cmd_queue.put(("query", bboxes_list))
        
        # Use asyncio.to_thread to avoid blocking the event loop
        # while waiting for results from the worker process
        return await asyncio.to_thread(self.worker.res_queue.get)
        
    def shutdown(self) -> None:
        """Gracefully shut down the worker process"""
        print("AsyncSpatialService: Shutting down worker process")
        self.worker.cmd_queue.put(("stop", None))
        self.worker.process.join(timeout=5)  # Wait up to 5 seconds for clean shutdown
        
        # Force terminate if it's still alive
        if self.worker.process.is_alive():
            print("AsyncSpatialService: Worker process didn't shut down gracefully, forcing termination")
            self.worker.process.terminate()