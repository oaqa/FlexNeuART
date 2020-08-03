package ciir.umass.edu.utilities;

public abstract class WorkerThread implements Runnable {
	protected int start = -1;
	protected int end = -1;
	public void set(int start, int end)
	{
		this.start = start;
		this.end = end;
	}
	public abstract WorkerThread clone();
}
