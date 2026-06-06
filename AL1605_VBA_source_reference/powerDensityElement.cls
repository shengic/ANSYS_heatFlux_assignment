' calculate power information for a single spectra grid element

Public elementNumber As Long
Public fourNodes As Collection
Public interpolateCoefficients As Collection
Public averagePowerDensity As Double
Public elementArea As Double
Public totalPower As Double
Private i As Long
Private j As Long
Private k As Long
Public m As Long
Public x1 As Double
Public x2 As Double
Public y1 As Double
Public y2 As Double

Public Sub initialized(ielement As Long, totalColumn As Long)
 m = totalColumn
 elementNumber = ielement
 Set fourNodes = New Collection
 Call putElementNodes
 Call calculcateInterpolateCoefficients
End Sub


' power element four nodes order
'
'       1-------3
'       |       |
'       |       |
'       |       |
'       2-------4
'
Public Sub putElementNodes()
 averagePowerDensity = 0#
 elementArea = 0#
 totalPower = 0#
 Dim ii As Long
 Dim j As Long
 Dim i As Long
 Dim powerDNode As powerDensityNode

' Set getElementNodes = New Collection
 ' j-th row
' m = spectraTotalColumn
 j = WorksheetFunction.Ceiling_Math(elementNumber / (m - 1))
 ' i-th column
 i = elementNumber - (j - 1) * (m - 1)


' Calculate indices
Dim index1 As Long, index2 As Long, index3 As Long, index4 As Long
index1 = (j - 1) * m + i
index2 = j * m + i
index3 = (j - 1) * m + i + 1
index4 = j * m + i + 1

' Add items with unique keys
fourNodes.Add Item:=SpectraPowerNodes.Item(index1), Key:="1"
fourNodes.Add Item:=SpectraPowerNodes.Item(index2), Key:="2"
fourNodes.Add Item:=SpectraPowerNodes.Item(index3), Key:="3"
fourNodes.Add Item:=SpectraPowerNodes.Item(index4), Key:="4"

 
 Dim pnode1 As powerDensityNode
 Dim pnode4 As powerDensityNode
 
 Set pnode1 = fourNodes.Item("1")
 Set pnode4 = fourNodes.Item("4")
 
'      (x1,y2)
'       1-------3
'       |       |
'       |       |
'       |       |
'       2-------4
'              (x2,y1)
 
 x1 = pnode1.xmrad
 y2 = pnode1.ymrad
 x2 = pnode4.xmrad
 y1 = pnode4.ymrad
 
 If x1 = x2 Then
  Debug.Print ("x1 and x2 are the same ! something is wrong")
  End
 End If
 
 If y1 = y2 Then
  Debug.Print ("y1 and y2 are the same ! something is wrong")
  End
 End If
 
 'calculate element Area (even they should be same)
  elementArea = Abs((x1 - x2) * (y1 - y2))
 'calculate average power density
 averagePowerDensity = 0#
 For ii = 1 To 4
  Set powerDNode = fourNodes.Item(ii)
  averagePowerDensity = averagePowreDensity + powerDNode.powerDensity / 4#
 Next ii

 averagePowerDensity = (fourNodes.Item(1).powerDensity + fourNodes.Item(2).powerDensity + _
                      fourNodes.Item(3).powerDensity + fourNodes.Item(4).powerDensity) / 4#

 ' calculate total power of this single element
  totalPower = averagePowerDensity * elementArea
 ' Debug.Print ("area = " & Format(elementArea, "0.0000"))
End Sub

' https://en.wikipedia.org/wiki/Bilinear_interpolation using polynomial fit

Public Sub calculcateInterpolateCoefficients()
  Dim matrixA(3, 3) As Double '4x4 array
  Dim matrixB(3) As Double    '4x1 array
  Dim matrixC(3, 3) As Double 'resulting 4x4 array
  Dim i As Integer

'populate matrixA with values
    matrixA(0, 0) = x2 * y2
    matrixA(0, 1) = -x2 * y1
    matrixA(0, 2) = -x1 * y2
    matrixA(0, 3) = x1 * y1
    
    matrixA(1, 0) = -y2
    matrixA(1, 1) = y1
    matrixA(1, 2) = y2
    matrixA(1, 3) = -y1
    
    matrixA(2, 0) = -x2
    matrixA(2, 1) = x2
    matrixA(2, 2) = x1
    matrixA(2, 3) = -x1
    
    matrixA(3, 0) = 1
    matrixA(3, 1) = -1
    matrixA(3, 2) = -1
    matrixA(3, 3) = 1
    
'populate matrixB with values
    matrixB(0) = fourNodes.Item("2").powerDensity / ((x2 - x1) * (y2 - y1))
    matrixB(1) = fourNodes.Item("1").powerDensity / ((x2 - x1) * (y2 - y1))
    matrixB(2) = fourNodes.Item("4").powerDensity / ((x2 - x1) * (y2 - y1))
    matrixB(3) = fourNodes.Item("3").powerDensity / ((x2 - x1) * (y2 - y1))
    
    'perform matrix multiplication
    For i = 0 To 3
        For j = 0 To 3
            matrixC(i, j) = 0
            For k = 0 To 3
                matrixC(i, j) = matrixC(i, j) + matrixA(i, k) * matrixB(k)
            Next
        Next
    Next
    
    Set interpolateCoefficients = New Collection
    For i = 0 To 3
     interpolateCoefficients.Add matrixC(i, i), CStr(i + 1)
    Next i
    
End Sub

' not use
Public Sub calculateAllElementNodes()
 'Set allElementNodes = New Collection
  Dim coll As Collection
 Dim elementNumber As Integer
 For elementNumber = 1 To totalElement
   Set coll = getElementNodes(elementNumber)
    coll.Add elementNumber, "spectra Element Number"
   allElementNodes.Add coll, CStr(elementNumber)
 Next elementNumber
End Sub

' not use
Public Function getAllElementNodes() As Collection
 Set getAllElementNodes = New Collection
 getAllElementNodes = allElementNodes
End Function