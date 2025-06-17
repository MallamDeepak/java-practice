public class dummy{
    public static void main(String[] args) {
        int num=2861;
        int count=0;
        for(int i=2;i<=num/2;i++){
            if(num%i==0){
                count++;
            }

        }
        if(count==1){
            System.out.println("Prime");
        }
        else{
            System.out.println("Not Prime");
        }

}}