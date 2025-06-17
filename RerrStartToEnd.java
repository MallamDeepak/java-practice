public class RerrStartToEnd{
    public static void main(String[] args) {
        printRange(0,6);

        
    }
    static void printRange(int start,int end){
        if(start>end){
            return;
        }
        System.out.println(start);
        printRange(start+1,end);

    }
    
}